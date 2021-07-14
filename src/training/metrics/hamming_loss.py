from typing import Optional

import torch
from allennlp.common.checks import ConfigurationError

from allennlp.training.metrics import Metric


@Metric.register("hamming_loss")
class HammingLoss(Metric):
    """Compute the average Hamming loss.

    The Hamming loss is the fraction of labels that are incorrectly predicted.
    """
    def __init__(self,
                 multi_label: bool = False) -> None:
        self._incorrect_count = 0.
        self._total_count = 0.
        self._multi_label = multi_label

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.detach_tensors(
            predictions, gold_labels, mask
        )

        if self._multi_label:
            batch_size = predictions.size(0)
            num_classes = predictions.size(-2)

            if mask is None:
                mask = torch.ones(batch_size)
            mask = mask.float().view(batch_size, -1)
            gold_labels = gold_labels.float()
            argmax_predictions = predictions.max(dim=-1)[1].float().squeeze(dim=-1)
            not_equal = (argmax_predictions != gold_labels).float() * mask
            self._incorrect_count += not_equal.sum().item()
            self._total_count += mask.sum().item() * num_classes
        else:
            batch_size = predictions.size(0)
            num_classes = predictions.size(-1)

            if (gold_labels >= num_classes).any():
                raise ConfigurationError("A gold label passed to HammingLoss contains "
                                         f"an id >= {num_classes}, the number of classes.")

            if mask is None:
                mask = torch.ones(batch_size)
            mask = mask.float()
            gold_labels = gold_labels.float()
            argmax_predictions = predictions.max(dim=-1)[1].float().squeeze(dim=-1)
            not_equal = (argmax_predictions != gold_labels).float() * mask
            self._incorrect_count += not_equal.sum().item()
            self._total_count += mask.sum().item()

    def get_metric(self,
                   reset: bool = False):
        """
        Returns
        -------
        The accumulated Hamming loss.
        """
        loss = float(self._incorrect_count) / float(self._total_count)
        if reset:
            self.reset()
        return loss

    def reset(self) -> None:
        self._incorrect_count = 0.0
        self._total_count = 0.0
