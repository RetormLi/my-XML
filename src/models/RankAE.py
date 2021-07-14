import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator

# from allennlp.training.metrics import BooleanAccuracy
from src.training.metrics import DiscountedCumulativeGainAtKMeasure
from src.training.metrics import FBetaAtKMeasure
from src.training.metrics import MultiLabelFBetaMeasure
from src.training.metrics import NormalizedDiscountedCumulativeGainAtKMeasure

logger = logging.getLogger(__name__)

from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow


# https://github.com/ssmele/C2AEinTorch/blob/master/C2AE.py

@Model.register('rankae')
class RankAE(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 feature_size: int,
                 label_size: int,
                 hidden_size: int,
                 latent_size: int,
                 reduction_ratio: int,
                 batch_size: int,
                 loss_lambda: float,
                 loss_m: float,
                 predict_threshold: float,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.label_size = label_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.loss_lambda = loss_lambda
        self.reduction_ratio = reduction_ratio
        self.loss_m = loss_m
        self.predict_threshold = predict_threshold
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.F = FeatureEmbedding(self.feature_size, self.hidden_size, self.latent_size, self.reduction_ratio).to(
            self.device)
        self.E = Encoder(label_size, hidden_size, latent_size).to(self.device)
        self.D = Decoder(latent_size, hidden_size, label_size).to(self.device)

        self.MSEloss = nn.MSELoss()
        self.x_h = torch.zeros(batch_size, latent_size).to(self.device)
        self.y_h = torch.zeros(batch_size, latent_size).to(self.device)

        self.macro_f1 = MultiLabelFBetaMeasure(average='macro')
        self.micro_f1 = MultiLabelFBetaMeasure(average='micro')
        self.fbeta_at_k = FBetaAtKMeasure(k=[1, 3, 5])
        self.dcg_at_k = DiscountedCumulativeGainAtKMeasure(k=[1, 3, 5])
        self.ndcg_at_k = NormalizedDiscountedCumulativeGainAtKMeasure(
            k=[1, 3, 5]
        )
        self.metrics = [
            # self.hamming_loss,
            # self.macro_f1,
            self.micro_f1,
            self.fbeta_at_k,
            # self.dcg_at_k,
            self.ndcg_at_k
        ]

        initializer(self)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        metrics.update({
            name: value
            for name, value in self.fbeta_at_k.get_metric(reset).items()
        })
        metrics.update({
            name: value
            for name, value in self.ndcg_at_k.get_metric(reset).items()
        })
        metrics.update({
            name: value
            for name, value in self.micro_f1.get_metric(reset).items()
        })
        # metrics.update({
        #     name: value
        #     for name, value in self.macro_f1.get_metric(reset).items()
        # })

        return metrics

    def forward(self,
                text: torch.LongTensor,
                label: torch.LongTensor = None,
                meta=None) -> Dict[str, torch.Tensor]:
        """
        forward step
        :param meta:
        :return: output_dict
        """
        # batch_size = len(text['tokens'])
        # num_tokens = len(text['tokens'][0])
        # Shape: (batch_size, num_tokens, embedding_dim)
        # tokens = self.text_field_embedder(text)
        # Shape: (batch_size, num_tokens)
        # mask = util.get_text_field_mask(text)

        # with torch.autograd.profiler.profile() as prof:
        batch_size = label.shape[1]
        has_label = 'labels' in meta[0]
        texts = None
        labels = None
        texts = text
        if has_label:
            labels = label

        output_dict = dict()

        x_h = None
        y_h = None

        if self.training:
            x_h = self.F(texts.float())
            y_h = self.E(labels.float())
            output_probs = self.D(y_h).to(self.device)
            # output_labels = metric_util.logits_to_predictions(output_probs, self.predict_threshold).float()
            output_labels = (output_probs > self.predict_threshold).long().to(self.device)
            if output_labels.ndim == 1:
                output_labels.unsqueeze_(0)
            output_dict = {"predictions": output_labels}

        else:
            x_h = self.F(texts)
            output_probs = self.D(x_h).to(self.device)
            # output_labels = metric_util.logits_to_predictions(output_probs, self.predict_threshold).float()
            output_labels = (output_probs > self.predict_threshold).long().to(self.device)
            if output_labels.ndim == 1:
                output_labels.unsqueeze_(0)
            output_dict = {"predictions": output_labels}

        if len(labels) != 0:
            for metric in self.metrics:
                # if metric == self.micro_f1:
                #     metric(output_labels, labels)
                # else:
                metric(output_labels, labels)

            if y_h is not None:
                output_dict["loss"] = self.get_loss(x_h, y_h, output_probs, output_labels, labels.float())

        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return output_dict

    def loss_h(self, x_h, y_h):
        """
        compute BCE loss
        :param x_h: batch * latent_size
        :param y_h: batch * latent_size
        :return:
        """
        return self.MSEloss(x_h, y_h)

    def loss_ae(self, y: torch.Tensor, probs: torch.Tensor, m):
        def loss_P(negative_y, positive_y):
            y_p = torch.min(positive_y)
            items = torch.relu(m + negative_y - y_p)
            return items.sum()

        def loss_N(negative_y, positive_y):
            y_n = torch.max(negative_y)
            items = torch.relu(m + y_n - positive_y)
            return items.sum()

        # y: batch*label_size
        negative_y = probs[y == 0]
        positive_y = probs[y == 1]

        return loss_P(negative_y, positive_y) + loss_N(negative_y, positive_y)

    def get_loss(self, x_h, y_h, outputs, labels,
                 targets, target_mask=None):
        return self.loss_h(x_h, y_h) + self.loss_lambda * self.loss_ae(targets, outputs, self.loss_m)


class FeatureEmbedding(torch.nn.Module):
    """
    Feature embedding step
    """

    def __init__(self, feature_size, hidden_size, out_size, reduction_ratio):
        super(FeatureEmbedding, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.fc = torch.nn.Linear(feature_size, out_size).to(self.device)
        self.emb_attn = DualAttention(feature_size, hidden_size, reduction_ratio).to(self.device)

    def forward(self, x):
        x = self.emb_attn(x)
        x = F.leaky_relu(self.fc(x))
        return x


class Encoder(torch.nn.Module):
    """
    Encoder
    """

    def __init__(self, label_size, hidden_size, out_size):
        super(Encoder, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.fc1 = nn.Linear(label_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, out_size).to(self.device)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x


class Decoder(torch.nn.Module):
    """
    Decoder
    """

    def __init__(self, latent_size, hidden_size, out_size):
        super(Decoder, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.fc1 = nn.Linear(latent_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, out_size).to(self.device)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(x)
        return x


class DualAttention(torch.nn.Module):
    """
    Dual Attention
    """

    def __init__(self, feature_size, hidden_size, reduction_ratio):
        super(DualAttention, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.attn_size = hidden_size // reduction_ratio
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.E = nn.Parameter(torch.randn(feature_size, hidden_size)).to(self.device)
        # self.E = nn.Linear(feature_size, hidden_size)
        self.F1 = nn.Linear(hidden_size, self.attn_size).to(self.device)
        self.F2 = nn.Linear(self.attn_size, hidden_size).to(self.device)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, V):
        # V: batch * feature_size
        # E: feature_size * hidden_size

        batch_size = V.shape[0]
        # expand_E: batch_size * feature_size * hidden_size
        expand_E = self.E.expand(batch_size, self.E.shape[0], self.E.shape[1])
        # batch * feature_size * hidden_size
        embedding = torch.mul(expand_E.permute(2, 0, 1), V).permute(1, 2, 0)
        # batch * feature_size * hidden_size
        attention = torch.sigmoid(self.F2(torch.relu(self.F1(embedding.float().to(self.device)))))
        # batch * feature_size * hidden_size
        rescale_emb = torch.mul(embedding, attention)
        # batch * feature_size
        feature_emb = self.pooling(rescale_emb).squeeze(dim=-1)

        return feature_emb
