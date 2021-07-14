import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric
from overrides import overrides
import torch

from src.training.metrics.util import prf_divide

logger = logging.getLogger(__name__)


@Metric.register('fbeta@k')
class FBetaAtKMeasure(Metric):
    def __init__(
            self,
            k: Union[int, List[int]],
            beta: float = 1.0,
    ):
        if not isinstance(k, List):
            k = [k]
        else:
            k = list(k)
        if any(item <= 0 for item in k):
            raise ConfigurationError(
                '`k` should be >0 in the F-beta@K score.'
            )
        if beta <= 0:
            raise ConfigurationError(
                '`beta` should be >0 in the F-beta score.'
            )
        self._k = k
        self._beta = beta

        # precision
        # int -> Tensor (Shape: ())
        self._precision: Union[None, Dict[int, torch.Tensor]] = None
        # recall
        # int -> Tensor (Shape: ())
        self._recall: Union[None, Dict[int, torch.Tensor]] = None
        # fbeta
        # int -> Tensor (Shape: ())
        self._fbeta: Union[None, Dict[int, torch.Tensor]] = None
        # the total number of instances
        # Shape: ()
        self._total_num: Union[None, torch.Tensor] = None

    @overrides
    def __call__(
            self,
            predicted_scores: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ):
        scores, gold_labels, mask = self.unwrap_to_tensors(
            predicted_scores, gold_labels, mask
        )

        # Calculate precision, recall, fbeta (instance-wise)
        batch_size, num_label_classes = predicted_scores.shape
        if gold_labels.size(-1) != num_label_classes:
            raise ConfigurationError(
                f'The `gold_labels` passed to {__class__} have an invalid '
                f'shape of {gold_labels.shape}, considering the number of '
                f'label classes is {num_label_classes}.'
            )

        device = predicted_scores.device
        # It means we call this metric at the first time
        # when `self._total_num` is None.
        if self._total_num is None:
            self._precision = {
                k: torch.tensor(0.0, device=device)
                for k in self._k
            }
            self._recall = {
                k: torch.tensor(0.0, device=device)
                for k in self._k
            }
            self._fbeta = {
                k: torch.tensor(0.0, device=device)
                for k in self._k
            }
            self._total_num = torch.tensor(0.0, device=device)

        if mask is None:
            mask = torch.ones(batch_size).bool()
        mask = mask.unsqueeze(dim=-1).bool().to(device)
        gold_labels = gold_labels.float().to(device)

        indices = torch.argsort(torch.argsort(
            predicted_scores, descending=True))
        for k in self._k:
            prediction = torch.ones_like(predicted_scores)
            prediction[indices + 1 > k] = 0.0
            pred_sum = (prediction.long() & mask.long()).sum(dim=-1)
            true_sum = (gold_labels.long() & mask.long()).sum(dim=-1)

            true_positives = prediction * gold_labels * mask
            true_positives = true_positives.sum(dim=-1)
            self._precision[k] += (true_positives / k).sum().to(torch.float)
            self._recall[k] += (true_positives / true_sum).sum()

            numerator = (1 + self._beta ** 2) * true_positives
            denominator = self._beta ** 2 * true_sum + pred_sum
            self._fbeta[k] += prf_divide(numerator, denominator).sum()
        self._total_num += mask.sum().to(torch.float)

    @overrides
    def reset(self) -> None:
        self._precision = None
        self._recall = None
        self._fbeta = None
        self._total_num = None

    @overrides
    def get_metric(
        self, reset: bool = False
    ):
        metric = {
            f'R@{k}': prf_divide(self._recall[k], self._total_num).item()
            for k in self._k
        }
        metric.update({
            f'P@{k}': prf_divide(self._precision[k], self._total_num).item()
            for k in self._k
        })
        metric.update({
            f'FBeta@{k}': prf_divide(self._fbeta[k], self._total_num).item()
            for k in self._k
        })

        if reset:
            self.reset()
        return metric
