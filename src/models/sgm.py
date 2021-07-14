import logging
from typing import Any, Dict, List, Tuple, Callable

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.attention import AdditiveAttention
from allennlp.nn import InitializerApplicator
from allennlp.nn import util
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.checks import ConfigurationError
from allennlp.models.encoder_decoders import SimpleSeq2Seq
from allennlp.nn.beam_search import BeamSearch

import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import LSTMCell
import torch.nn.functional as F

# from allennlp.training.metrics import BooleanAccuracy
from src.training.metrics import DiscountedCumulativeGainAtKMeasure
from src.training.metrics import FBetaAtKMeasure
from src.training.metrics import MultiLabelFBetaMeasure
from src.training.metrics import NormalizedDiscountedCumulativeGainAtKMeasure
from src.training.metrics import MultiLabelHammingLoss

import warnings

logger = logging.getLogger(__name__)
from collections import namedtuple


@Model.register('sgm')
class SGM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder_num_layers: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int,
                 decoder_num_layers: int,
                 dropout: float,
                 batch_size: int,
                 bottleneck_size: int,
                 max_time_step: int,
                 beam_size: int,
                 target_embedding_dim: int,
                 target_namespace: str = "label_tokens",
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self._target_namespace = target_namespace
        self.label_size = self.vocab.get_vocab_size(target_namespace)
        self.embedding_size = self.text_field_embedder.get_output_dim()
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.batch_size = batch_size
        self.max_time_step = max_time_step
        self.target_embedding_dim = target_embedding_dim
        self.bottleneck_size = bottleneck_size
        self.beam_size = beam_size
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._start_index = self.vocab.get_token_index(
            START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(
            END_SYMBOL, self._target_namespace)

        self.target_embedder = Embedding(
            self.label_size, self.target_embedding_dim)
        # self._encoder = encoder
        self._encoder = nn.LSTM(self.embedding_size, self.encoder_hidden_size,
                                self.encoder_num_layers, batch_first=True,
                                bidirectional=True, dropout=dropout)

        # self._decoder_cell = LSTMCell(self.target_embedding_dim + 2 * self.encoder_hidden_size,
        #                               self.decoder_hidden_size)
        self.decoder = nn.LSTM(self.target_embedding_dim + 2 * self.encoder_hidden_size,
                               self.decoder_hidden_size, self.decoder_num_layers,
                               bidirectional=False, dropout=dropout)
        self.dropout = Dropout(dropout)
        self.attention = AdditiveAttention(
            self.decoder_hidden_size, 2 * self.encoder_hidden_size, normalize=False)

        self.input2attn = nn.Linear(self.decoder_hidden_size + 2 * self.encoder_hidden_size,
                                    self.decoder_hidden_size)
        self.attn2score = nn.Linear(self.decoder_hidden_size, 1)

        self.GE_W2 = nn.Linear(self.target_embedding_dim,
                               self.target_embedding_dim)
        self.GE_W1 = nn.Linear(self.target_embedding_dim,
                               self.target_embedding_dim)
        self.context_hidden_combinator = nn.Linear(self.decoder_hidden_size + 2 * self.encoder_hidden_size,
                                                   self.bottleneck_size)
        self.output_trans = nn.Linear(self.bottleneck_size, self.label_size)

        torch.nn.init.xavier_normal_(self.input2attn.weight)
        torch.nn.init.xavier_normal_(self.attn2score.weight)
        torch.nn.init.xavier_normal_(self.GE_W1.weight)
        torch.nn.init.xavier_normal_(self.GE_W2.weight)
        torch.nn.init.xavier_normal_(self.context_hidden_combinator.weight)
        torch.nn.init.xavier_normal_(self.output_trans.weight)

        # self._beam_search = SgmBeamSearch(
        #     self._end_index, max_steps=max_time_step, beam_size=beam_size)
        # self._beam_search = BeamSearch(
        #     self._end_index, max_steps=max_time_step, beam_size=beam_size)

        # self.bool_acc = BooleanAccuracy()
        # self.macro_f1 = MultiLabelFBetaMeasure(average='macro')
        self.micro_f1 = MultiLabelFBetaMeasure(average='micro')
        self.ml_hm_loss = MultiLabelHammingLoss()
        self.fbeta_at_k = FBetaAtKMeasure(k=[1, 3, 5])
        self.dcg_at_k = DiscountedCumulativeGainAtKMeasure(k=[1, 3, 5])
        self.ndcg_at_k = NormalizedDiscountedCumulativeGainAtKMeasure(
            k=[1, 3, 5]
        )
        self.metrics = [
            self.ml_hm_loss,
            # self.hamming_loss,
            # self.macro_f1,
            self.micro_f1,
            self.fbeta_at_k,
            # self.dcg_at_k,
            self.ndcg_at_k
        ]
        initializer(self)

        self.Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                labels: Dict[str, torch.LongTensor] = None,
                meta: Dict[str, Any] = None) -> Dict[str, torch.Tensor]:
        # state: Dict with keys:
        # "source_mask": (batch_size, max_input_sequence_length),
        # "encoder_outputs": (batch_size, max_input_sequence_length, encoder_output_dim),
        # "target_mask":  (batch_size, output_sequence_length)
        encode_state = self._encode(text, labels)
        self.batch_size = encode_state["source_mask"].size(0)

        output_dict = dict()
        if labels is not None:
            # add "decode_hidden" and "decode_context" to state for LSTM input
            init_state = self._init_decoder_state(encode_state.copy())
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(init_state, labels)

        # [:, 1:] for removing start label
        target_mask = encode_state["target_mask"][:, 1:]
        target_labels = labels["tokens"][:, 1:]
        onehot_label = self.one_hot_predict(target_labels, target_mask)

        if not self.training:
            # state: Dict with: "source_mask", "encoder_outputs", "target_mask", "decode_hidden"
            # "decode_context", "attended_context"
            state = self._init_decoder_state(encode_state.copy())

            # "predictions": (batch_size, max_time_steps),
            batch_size = state["source_mask"].size(0)
            predictions = []

            for i in range(batch_size):
                new_state = self.get_state_from_index(state, i)
                predictions.append(self.beam_search(new_state, self.beam_size, self.max_time_step))
            predictions = torch.tensor(predictions).to(self.device)
            output_dict["predictions"] = predictions
            # transform token expression into onehot expression
            onehot_pred = self.one_hot_predict(predictions)
            for metric in self.metrics:
                metric(onehot_pred, onehot_label)

        else:
            onehot_pred = self.one_hot_predict(output_dict["predictions"])
            # remove unk, start, end
            for metric in self.metrics:
                metric(onehot_pred, onehot_label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        metrics.update({
            'multi_label_hamming_loss': self.ml_hm_loss.get_metric(reset)
        })
        metrics.update({
            name: value
            for name, value in self.micro_f1.get_metric(reset).items()
        })
        metrics.update({
            name: value
            for name, value in self.fbeta_at_k.get_metric(reset).items()
        })
        metrics.update({
            name: value
            for name, value in self.ndcg_at_k.get_metric(reset).items()
        })
        return metrics

    def one_hot_predict(self, predictions, mask=None):
        # predictions: (batch_size, max_time_step)
        onehot = torch.zeros(
            predictions.shape[0], self.label_size).to(self.device)
        if mask is not None:
            # remove padding, unk, start, end
            return onehot.scatter_(1, predictions * mask, 1)[:, 4:]
        else:
            # remove padding, unk, start, end
            return onehot.scatter_(1, predictions, 1)[:, 4:]

    def _encode(self, source_tokens: Dict[str, torch.Tensor],
                target_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self.text_field_embedder(source_tokens)
        batch_size = embedded_input.size(0)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, output_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        # hn: (2*layers, batch, hidden_size)
        # cn: (2*layers, batch, hidden_size)
        encoder_outputs, (hn, cn) = self._encoder(embedded_input)
        return {
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs,
            "target_mask": target_mask,
            "encoder_h": hn,
            "encoder_c": cn
        }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # Initialize the decoder hidden state with the final output of the encoder.
        if 2 * self.encoder_hidden_size == self.decoder_hidden_size \
                and self.encoder_num_layers >= self.decoder_num_layers:
            # shape: (2 * num_layers, batch_size, decoder_output_dim)
            state["encoder_h"] = state["encoder_h"].view(self.encoder_num_layers, 2, batch_size,
                                                         self.encoder_hidden_size)
            state["encoder_h"] = state["encoder_h"][-self.decoder_num_layers:, :, :, :].transpose(1, 2)
            state["encoder_h"] = state["encoder_h"].reshape(self.decoder_num_layers, batch_size,
                                                            self.decoder_hidden_size).to(self.device)
            state["decoder_hidden"] = state["encoder_h"].transpose(0, 1)
            state["encoder_c"] = state["encoder_c"].view(self.encoder_num_layers, 2, batch_size,
                                                         self.encoder_hidden_size)
            state["encoder_c"] = state["encoder_c"][-self.decoder_num_layers:, :, :, :].transpose(1, 2)
            state["encoder_c"] = state["encoder_c"].reshape(self.decoder_num_layers, batch_size,
                                                            self.decoder_hidden_size).to(self.device)
            state["decoder_context"] = state["encoder_c"].transpose(0, 1)
        else:
            state["decoder_hidden"] = torch.zeros(
                batch_size, self.decoder_num_layers, self.decoder_hidden_size).float().to(self.device)
            # shape: (num_layers, batch_size, decoder_output_dim)
            state["decoder_context"] = torch.zeros(
                batch_size, self.decoder_num_layers, self.decoder_hidden_size).float().to(self.device)

        state["decoder_output"] = torch.zeros(
            batch_size, 1, self.decoder_hidden_size).float().to(self.device)
        state["attended_context"] = torch.zeros(
            batch_size, 1, 2 * self.encoder_hidden_size).float().to(self.device)
        state["class_probabilities"] = torch.zeros(
            batch_size, self.label_size).float().to(self.device)
        state["class_probabilities"][:, self._start_index] = 1

        del state["encoder_h"], state["encoder_c"]

        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        target_mask = state["target_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            targets = None
            num_decoding_steps = self.max_time_step

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = target_mask.new_full(
            (batch_size,), fill_value=self._start_index)

        class_probabilities = torch.zeros(
            batch_size, self.label_size).float().to(self.device)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(
                input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            # class_probabilities = F.softmax(output_projections, dim=-1)
            class_probabilities = state["class_probabilities"]

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            loss = self._get_loss(logits.float(), targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, max_output_sequence_length)
        target_mask = state["target_mask"]

        # shape: (numlayers, group_size, decoder_output_dim)
        # transpose for the input of torch.LSTM
        decoder_hidden = state["decoder_hidden"].transpose(0, 1)

        # shape: (numlayers, group_size, decoder_output_dim)
        decoder_context = state["decoder_context"].transpose(0, 1)

        # shape: (1, group_size, decoder_output_dim)
        decoder_output = state["decoder_output"].transpose(0, 1).squeeze(0)

        # shape: (1, batch_size, decoder_hidden_size)
        attended_context = state["attended_context"].transpose(0, 1)

        # shape: (batch_size, label_size)
        class_probabilities = state["class_probabilities"]

        # shape: (batch_size, target_embedding_dim)
        embedded_input = self.target_embedder(last_predictions)

        # # output shape: (group_size, encoder_output_dim) = (b, 2 * encoder_hidden_size)
        # # c_t:
        # attended_input = self._prepare_attended_input(
        #     decoder_output, encoder_outputs, source_mask)

        # shape: (b, target_embed_dim)
        global_embed = self.global_embedding(
            embedded_input, class_probabilities)

        # shape: (1, b, 2 * hidden_size + target_embed_dim)
        decoder_input = torch.cat((attended_context, global_embed.unsqueeze(0)), dim=-1)

        # # the input of LSTM layer requires 3 dims
        # input shape: (1, batch, input_size)
        # output shape: (1, batch, decoder_hidden_size)
        decoder_output, (decoder_hidden, decoder_context) = self.decoder(
            decoder_input.contiguous(),
            (decoder_hidden.contiguous(), decoder_context.contiguous()))

        # shape: (1, batch_size, encoder_output_dim)
        attended_input = self._prepare_attended_input(
            decoder_output, encoder_outputs, source_mask)
        # shape: (group_size, label_size)
        output_projections = self.output_layer(
            decoder_output, attended_input)

        state["decoder_hidden"] = decoder_hidden.transpose(0, 1)
        state["decoder_context"] = decoder_context.transpose(0, 1)
        state["decoder_output"] = decoder_output.transpose(0, 1)
        state["attended_context"] = attended_input.transpose(0, 1)
        state["class_probabilities"] = F.softmax(output_projections, dim=-1)

        return output_projections, state

    def global_embedding(self, embedded_input, prev_prob):
        batch_size = embedded_input.size(0)
        # shape： (label_size, target_embedding_size)
        label_embedding = self.target_embedder.weight
        if prev_prob is None:
            prev_prob = torch.zeros(
                batch_size, self.label_size).float().to(self.device)
        emb_avg = torch.matmul(prev_prob, label_embedding)
        H = torch.sigmoid(self.GE_W1(embedded_input) + self.GE_W2(emb_avg))
        global_embs = (1 - H) * embedded_input + H * emb_avg

        return global_embs

    def output_layer(self, hidden_state, context):
        bottleneck_vec = torch.relu(self.context_hidden_combinator(
            torch.cat((hidden_state, context), dim=-1).squeeze(0)
        ))
        bottleneck_vec = self.dropout(bottleneck_vec)
        output_vec = self.output_trans(bottleneck_vec)

        return output_vec

    def _prepare_attended_input(self,
                                decoder_hidden_state,
                                encoder_outputs,
                                masks) -> torch.Tensor:
        # input (1, batch, decoder_output_size)
        expand_output = decoder_hidden_state.expand(encoder_outputs.size()[1],
                                                    decoder_hidden_state.size()[1],
                                                    decoder_hidden_state.size()[2]).transpose(0, 1)
        # (batch,seq_length,decoder_hid)

        attn_input = torch.cat(
            (expand_output, encoder_outputs), dim=-1)  # (b,s_l,dec+2*enc)

        attn_output = torch.tanh(self.input2attn(
            attn_input))  # (b,a_l,attn_size)
        score = self.attn2score(attn_output).squeeze(2)  # (b,s_l)

        # score = self.attention(
        #     decoder_hidden_state, encoder_outputs)

        # mask the input sequence.
        if masks is not None:
            score = score.masked_fill(~masks.bool(), -float('inf'))

        weights = F.softmax(score, dim=1).unsqueeze(1)

        # shape: (batch_size, 1, encoder_output_dim)
        attended_input = torch.bmm(weights, encoder_outputs)

        # shape: (1, batch_size, encoder_output_dim)
        return attended_input.transpose(0, 1)

    def get_state_from_index(self, state, index):
        new_state = dict()
        if type(index) == int:
            for key, state_tensor in state.items():
                # _, *last_dims = state_tensor.size()
                # shape: (batch_size * beam_size, *)
                new_state[key] = state_tensor[index].unsqueeze(0)
        # tensor
        else:
            for key, state_tensor in state.items():
                # _, *last_dims = state_tensor.size()
                # shape: (batch_size * beam_size, *)
                new_state[key] = state_tensor[index]
        return new_state

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]
                  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_probabilities, state

    @staticmethod
    def _get_loss(logits: torch.FloatTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets.long(), relevant_mask.float())

    # def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     """Make forward pass during prediction using a beam search."""
    #     batch_size = state["source_mask"].size()[0]
    #     start_predictions = state["source_mask"].new_full(
    #         (batch_size,), fill_value=self._start_index)
    #
    #     # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
    #     # shape (log_probabilities): (batch_size, beam_size)
    #     all_top_k_predictions, log_probabilities = self._beam_search.search(
    #         start_predictions, state, self.take_step)
    #
    #     output_dict = {
    #         "predictions": all_top_k_predictions[:, 0, :]
    #     }
    #     return output_dict

    def beam_search(self, state, beam_size: int = 5, max_decoding_time_step: int = 10):

        # prev_h = torch.zeros(self.decoder_num_layers, 1, self.decoder_hidden_size).to(self.device)
        # prev_c = torch.zeros(self.decoder_num_layers, 1, self.decoder_hidden_size).to(self.device)

        # encoder_output = state["encoder_outputs"]  # (1,seq_lem,2*enc_hid)

        # TODO inf?
        # prev_logits = state["class_probabilities"]
        # label_mask = torch.ones(1, self.label_size).long().to(self.device)
        # label_mask[:, self._start_index] = 0
        # prev_logits.masked_fill_(label_mask.byte(), -float('inf'))

        # prev_context = state["attended_context"]

        hypotheses = [[self._start_index]]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float).to(self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)  # current hyp cnt
            # expand_encoder_output = encoder_output.expand(hyp_num,
            #                                               encoder_output.size(1),
            #                                               encoder_output.size(2))
            # expand_prev_context = prev_context.expand(hyp_num,
            #                                           prev_context.size(1),
            #                                           prev_context.size(2))

            for key, value in state.items():
                _, *last_dims = value.size()
                state[key] = value.expand(hyp_num, *last_dims)

            prev_labels = torch.LongTensor([hyp[-1] for hyp in hypotheses]).to(self.device)
            # cur_input = self.label_embedding(prev_labels)  # (beam,label_emb)

            # cur_output, cur_context, (cur_h, cur_c), logits = \
            #     self.step(cur_input, expand_prev_context, (prev_h, prev_c), expand_encoder_output, None, prev_logits)

            # log probabilities over target words
            log_p_t, state = self.take_step(prev_labels, state)  # (b,label_space)

            live_hyp_num = beam_size - len(completed_hypotheses)

            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.sort(contiuating_hyp_scores, descending=True)

            prev_hyp_ids = top_cand_hyp_pos // self.label_size
            hyp_word_ids = top_cand_hyp_pos % self.label_size

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            k = 0

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                if k >= live_hyp_num:
                    break

                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()
                if hyp_word_id in hypotheses[prev_hyp_id]:
                    continue
                k += 1

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word_id]
                if hyp_word_id == self._end_index:
                    completed_hypotheses.append(self.Hypothesis(value=new_hyp_sent[1:-1],
                                                                score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids).to(self.device)
            state = self.get_state_from_index(state, live_hyp_ids)
            # prev_h, prev_c = cur_h[:, live_hyp_ids], cur_c[:, live_hyp_ids]
            # prev_logits = log_p_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.Tensor(new_hyp_scores).to(self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(self.Hypothesis(value=hypotheses[0][1:],
                                                        score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score / (len(hyp.value) + 1e-10), reverse=True)

        return completed_hypotheses[0].value

# class SgmBeamSearch:
#
#     def __init__(self,
#                  end_index: int,
#                  max_steps: int = 50,
#                  beam_size: int = 10,
#                  per_node_beam_size: int = None) -> None:
#         self._end_index = end_index
#         self.max_steps = max_steps
#         self.beam_size = beam_size
#         self.per_node_beam_size = per_node_beam_size or beam_size
#
#     def search(self,
#                start_predictions: torch.Tensor,
#                start_state: Dict[str, torch.Tensor],
#                start_prob: torch.Tensor,
#                step: Callable[[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
#                               Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) \
#             -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns
#         -------
#         Tuple[torch.Tensor, torch.Tensor]
#             Tuple of ``(predictions, log_probabilities)``, where ``predictions``
#             has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
#             has shape ``(batch_size, beam_size)``.
#         """
#         batch_size = start_predictions.size()[0]
#
#         # List of (batch_size, beam_size) tensors. One for each time step. Does not
#         # include the start symbols, which are implicit.
#         predictions: List[torch.Tensor] = []
#
#         # List of (batch_size, beam_size) tensors. One for each time step. None for
#         # the first.  Stores the index n for the parent prediction, i.e.
#         # predictions[t-1][i][n], that it came from.
#         backpointers: List[torch.Tensor] = []
#
#         # Calculate the first timestep. This is done outside the main loop
#         # because we are going from a single decoder input (the output from the
#         # encoder) to the top `beam_size` decoder outputs. On the other hand,
#         # within the main loop we are going from the `beam_size` elements of the
#         # beam to `beam_size`^2 candidates from which we will select the top
#         # `beam_size` elements for the next iteration.
#         # shape: (batch_size, num_classes)
#         start_class_probabilities, state = step(
#             start_predictions, start_state, start_prob)
#
#         num_classes = start_class_probabilities.size()[1]
#
#         # Make sure `per_node_beam_size` is not larger than `num_classes`.
#         if self.per_node_beam_size > num_classes:
#             raise ConfigurationError(f"Target vocab size ({num_classes:d}) too small "
#                                      f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
#                                      f"Please decrease beam_size or per_node_beam_size.")
#
#         # shape: (batch_size, beam_size), (batch_size, beam_size)
#         start_top_probabilities, start_predicted_classes = \
#             start_class_probabilities.topk(self.beam_size)
#         if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
#             warnings.warn("Empty sequences predicted. You may want to increase the beam size or ensure "
#                           "your step function is working properly.",
#                           RuntimeWarning)
#             return start_predicted_classes.unsqueeze(-1), start_top_probabilities
#
#         # The log probabilities for the last time step.
#         # shape: (batch_size, beam_size)
#         last_top_probabilities = start_top_probabilities
#
#         # shape: [(batch_size, beam_size)]
#         predictions.append(start_predicted_classes)
#
#         # Log probability tensor that mandates that the end token is selected.
#         # shape: (batch_size * beam_size, num_classes)
#         probs_after_end = start_class_probabilities.new_full(
#             (batch_size * self.beam_size, num_classes),
#             float("-inf")
#         )
#         probs_after_end[:, self._end_index] = 0.
#
#         # Set the same state for each element in the beam.
#         for key, state_tensor in state.items():
#             _, *last_dims = state_tensor.size()
#             # shape: (batch_size * beam_size, *)
#             state[key] = state_tensor. \
#                 unsqueeze(1). \
#                 expand(batch_size, self.beam_size, *last_dims). \
#                 reshape(batch_size * self.beam_size, *last_dims)
#
#         # shape: (batch_size, num_classes) -> (batch_size * beam_size, num_classes)
#         class_probabilities = start_class_probabilities.repeat(1, self.beam_size) \
#             .reshape(batch_size * self.beam_size, -1)
#         # shape: (1, batch_size, decoder_size) -> (1, batch_size * beam_size, decoder_size)
#         for timestep in range(self.max_steps - 1):
#             # shape: (batch_size * beam_size,)
#             last_predictions = predictions[-1].reshape(
#                 batch_size * self.beam_size)
#
#             # If every predicted token from the last step is `self._end_index`,
#             # then we can stop early.
#             if (last_predictions == self._end_index).all():
#                 break
#
#             # Take a step. This get the predicted log probs of the next classes
#             # and updates the state.
#             # shape: (batch_size * beam_size, num_classes)
#             class_probabilities = torch.exp(class_probabilities)
#             class_probabilities, state = step(
#                 last_predictions, state, class_probabilities)
#
#             # shape: (batch_size * beam_size, num_classes)
#             last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
#                 batch_size * self.beam_size,
#                 num_classes
#             )
#
#             # Here we are finding any beams where we predicted the end token in
#             # the previous timestep and replacing the distribution with a
#             # one-hot distribution, forcing the beam to predict the end token
#             # this timestep as well.
#             # shape: (batch_size * beam_size, num_classes)
#             cleaned_log_probabilities = torch.where(
#                 last_predictions_expanded == self._end_index,
#                 probs_after_end,
#                 class_probabilities
#             )
#
#             # shape (both): (batch_size * beam_size, per_node_beam_size)
#             top_log_probabilities, predicted_classes = \
#                 cleaned_log_probabilities.topk(self.per_node_beam_size)
#
#             # Here we expand the last log probabilities to (batch_size * beam_size, per_node_beam_size)
#             # so that we can add them to the current log probs for this timestep.
#             # This lets us maintain the log probability of each element on the beam.
#             # shape: (batch_size * beam_size, per_node_beam_size)
#             expanded_last_log_probabilities = last_top_probabilities. \
#                 unsqueeze(2). \
#                 expand(batch_size, self.beam_size, self.per_node_beam_size). \
#                 reshape(batch_size * self.beam_size, self.per_node_beam_size)
#
#             # shape: (batch_size * beam_size, per_node_beam_size)
#             summed_top_log_probabilities = top_log_probabilities + \
#                                            expanded_last_log_probabilities
#
#             # shape: (batch_size, beam_size * per_node_beam_size)
#             reshaped_summed = summed_top_log_probabilities. \
#                 reshape(batch_size, self.beam_size * self.per_node_beam_size)
#
#             # shape: (batch_size, beam_size * per_node_beam_size)
#             reshaped_predicted_classes = predicted_classes. \
#                 reshape(batch_size, self.beam_size * self.per_node_beam_size)
#
#             # Keep only the top `beam_size` beam indices.
#             # shape: (batch_size, beam_size), (batch_size, beam_size)
#             restricted_beam_probs, restricted_beam_indices = reshaped_summed.topk(
#                 self.beam_size)
#
#             # Use the beam indices to extract the corresponding classes.
#             # shape: (batch_size, beam_size)
#             restricted_predicted_classes = reshaped_predicted_classes.gather(
#                 1, restricted_beam_indices)
#
#             predictions.append(restricted_predicted_classes)
#
#             # shape: (batch_size, beam_size)
#             last_top_probabilities = restricted_beam_probs
#
#             # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
#             # indices with a common ancestor are grouped together. Hence
#             # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
#             # division as the tensor is a LongTensor.)
#             # shape: (batch_size, beam_size)
#             backpointer = restricted_beam_indices / self.per_node_beam_size
#
#             backpointers.append(backpointer)
#
#             # Keep only the pieces of the state tensors corresponding to the
#             # ancestors created this iteration.
#             for key, state_tensor in state.items():
#                 _, *last_dims = state_tensor.size()
#                 # shape: (batch_size, beam_size, *)
#                 expanded_backpointer = backpointer. \
#                     view(batch_size, self.beam_size, *([1] * len(last_dims))). \
#                     expand(batch_size, self.beam_size, *last_dims)
#
#                 # shape: (batch_size * beam_size, *)
#                 state[key] = state_tensor. \
#                     reshape(batch_size, self.beam_size, *last_dims). \
#                     gather(1, expanded_backpointer). \
#                     reshape(batch_size * self.beam_size, *last_dims)
#
#         if not torch.isfinite(last_top_probabilities).all():
#             warnings.warn("Infinite log probabilities encountered. Some final sequences may not make sense. "
#                           "This can happen when the beam size is larger than the number of valid (non-zero "
#                           "probability) transitions that the step function produces.",
#                           RuntimeWarning)
#
#         # Reconstruct the sequences.
#         # shape: [(batch_size, beam_size, 1)]
#         reconstructed_predictions = [predictions[-1].unsqueeze(2)]
#
#         # shape: (batch_size, beam_size)
#         cur_backpointers = backpointers[-1]
#
#         for timestep in range(len(predictions) - 2, 0, -1):
#             # shape: (batch_size, beam_size, 1)
#             cur_preds = predictions[timestep].gather(
#                 1, cur_backpointers).unsqueeze(2)
#
#             reconstructed_predictions.append(cur_preds)
#
#             # shape: (batch_size, beam_size)
#             cur_backpointers = backpointers[timestep -
#                                             1].gather(1, cur_backpointers)
#
#         # shape: (batch_size, beam_size, 1)
#         final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
#
#         reconstructed_predictions.append(final_preds)
#
#         # shape: (batch_size, beam_size, max_steps)
#         all_predictions = torch.cat(
#             list(reversed(reconstructed_predictions)), 2)
#
#         return all_predictions, last_top_probabilities
