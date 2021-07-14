from typing import Optional

from allennlp.training.metrics import Metric
import torch


@Metric.register('multi_label_hamming_loss')
class MultiLabelHammingLoss(Metric):
    """Compute the average hamming loss.
    The Hamming Loss is the fraction of labels that are incorrectly predicted.
    """
    def __init__(self) -> None:
        self._incorrect_count = 0.0
        self._total_count = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        parameter
        ---------
        predictions : `torch.Tensor`, required
            A tensor of predictions of shape (batch_size, num_label_classes).
        gold_labels : `torch.Tensor`, required
            A tensor of integer class label of shape (batch_size,
            num_label_classes). It must be the same shape as the
            `predictions` tensor.
        mask : `torch.Tensor`, optional (default = None).
            A masking tensor, whose shape must be (batch_size, ).
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(
            predictions, gold_labels, mask
        )

        batch_size, num_classes = predictions.shape
        if mask is None:
            mask = torch.ones(batch_size).to(gold_labels.device)
        mask = mask.view(batch_size, -1).bool()
        not_equal = (predictions.long() != gold_labels.long()) & mask.bool()
        self._incorrect_count += not_equal.sum().item()
        self._total_count += mask.sum().item() * num_classes

    def get_metric(self, reset: bool = False) -> float:
        """
        Returns
        -------
        The accumulated Hamming loss.
        """
        try:
            loss = self._incorrect_count / self._total_count
        except ZeroDivisionError:
            loss = 0.0
        if reset:
            self.reset()
        return loss

    def reset(self) -> None:
        self._incorrect_count = 0.0
        self._total_count = 0.0