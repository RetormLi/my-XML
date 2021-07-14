from overrides import overrides
import torch
from typing import List
from typing import Optional
from typing import Union

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric
from src.training.metrics.util import prf_divide


@Metric.register("multi_label_fbeta")
class MultiLabelFBetaMeasure(Metric):
    def __init__(
            self,
            beta: float = 1.0,
            average: str = None,
            labels: List[int] = None
    ) -> None:
        average_options = {None, 'micro', 'macro', 'weighted'}
        if average not in average_options:
            raise ConfigurationError(
                f'`average` has to be one of {average_options}.'
            )
        if beta <= 0:
            raise ConfigurationError(
                '`beta` should be >0 in the F-beta score.'
            )
        if labels is not None and len(labels) == 0:
            raise ConfigurationError(
                '`labels` cannot be an empty list.'
            )
        self._beta = beta
        self._average = average
        self._labels = labels

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_label_classes, )
        self._true_positive_sum: Union[None, torch.Tensor] = None
        # the total number of instances
        # Shape: (num_label_classes, )
        self._total_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_label_classes, )
        self._pred_sum: Union[None, torch.Tensor] = None
        # the total number of instance under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_label_classes, )
        self._true_sum: Union[None, torch.Tensor] = None

    @overrides
    def __call__(
            self,
            predictions: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        """
        # Parameters

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

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_label_classes = predictions.size(-1)
        if gold_labels.size(-1) != num_label_classes:
            raise ConfigurationError(
                f'The `gold_labels` passed to {__class__} have an invalid '
                f'shape {gold_labels.shape}, considering the number of '
                f'label classes is {num_label_classes}.'
            )

        device = predictions.device
        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(
                num_label_classes, device=device
            )
            self._true_sum = torch.zeros(
                num_label_classes, device=device
            )
            self._pred_sum = torch.zeros(
                num_label_classes, device=device
            )
            self._total_sum = torch.zeros(
                num_label_classes, device=device
            )

        if mask is None:
            mask = torch.ones_like(gold_labels).bool().to(device)
        else:
            batch_size = gold_labels.shape[0]
            mask = mask.view(batch_size, -1).bool().to(device)
        gold_labels = gold_labels.float().to(device)

        true_positives = (gold_labels == predictions) & (gold_labels == 1)
        true_positives = true_positives & mask

        # shape: (batch_size, ..., num_label_classes)
        true_positive_sum = true_positives.sum(-2)
        pred_sum = (predictions.long() & mask.long()).sum(-2)
        true_sum = (gold_labels.long() & mask.long()).sum(-2)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum(0).to(torch.float)

    @overrides
    def get_metric(
        self, reset: bool = False
    ):
        """
        # Returns

        precisions : `List[float]`
        recalls : `List[float]`
        f1-measures : `List[float]`

        !!! Note
            If `self.average` is not `None`, you will get `float` instead
            of `List[float]`.
        """
        if self._true_positive_sum is None:
            raise RuntimeError(
                'You never call this metric before.'
            )

        tp_sum = self._true_positive_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum

        if self._labels is not None:
            # Retain only selected labels and order them
            tp_sum = tp_sum[self._labels]
            pred_sum = pred_sum[self._labels]
            true_sum = true_sum[self._labels]

        if self._average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = prf_divide(tp_sum, pred_sum)
        recall = prf_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (
                beta2 * precision + recall
        )
        fscore[tp_sum == 0] = 0.0

        if self._average == 'macro':
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
        elif self._average == 'weighted':
            weights = true_sum
            weights_sum = true_sum.sum()
            precision = prf_divide(
                (weights * precision).sum(), weights_sum
            )
            recall = prf_divide(
                (weights * recall).sum(), weights_sum
            )
            fscore = prf_divide(
                (weights_sum * fscore).sum(), weights_sum
            )

        if reset:
            self.reset()

        if self._average is None:
            return {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'fscore': fscore.tolist()
            }
        else:
            return {
                'precision': precision.item(),
                'recall': recall.item(),
                'fscore': fscore.item()
            }

    @overrides
    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = (
                self._total_sum - self._pred_sum - self._true_sum +
                self._true_positive_sum
            )
            return true_negative_sum

    @property
    def true_sum(self) -> Optional[List[int]]:
        return self._true_sum

    @property
    def pred_sum(self) -> Optional[List[int]]:
        return self._pred_sum

    @property
    def true_positive_sum(self) -> Optional[List[int]]:
        return self._true_positive_sum

    @staticmethod
    def cast_as_indicator(prediction_scores: torch.Tensor,
                          threshold: float):
        return (prediction_scores >= threshold).long()
