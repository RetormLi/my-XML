from typing import Optional

from overrides import overrides
from typing import Dict

import torch
from typing import Union

from typing import List

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric
from src.training.metrics.util import prf_divide


@Metric.register('DCG@K')
class DiscountedCumulativeGainAtKMeasure(Metric):
    def __init__(
            self,
            k: Union[int, List[int]]
    ) -> None:
        if not isinstance(k, List):
            k = [k]
        else:
            k = list(k)
        self._k = k

        # Statistics
        # discounts for different ranks
        # Shape: (num_label_classes, )
        self._discounts: Union[None, torch.Tensor] = None
        # dcg -> Discounted Cumulative Gain
        # int -> Tensor (Shape: ())
        self._dcg: Union[None, Dict[int, torch.Tensor]] = None
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

        # Calculate dcg (Discounted Cumulative Gain), true_sum
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
            self._discounts = 1 / torch.log2(
                torch.arange(1, num_label_classes+1, device=device) + 1.0
            )
            self._dcg = {
                k: torch.tensor(0.0, device=device)
                for k in self._k
            }
            self._total_num = torch.tensor(0.0, device=device)

        if mask is None:
            mask = torch.ones(batch_size).bool()
        mask = mask.unsqueeze(dim=-1).bool().to(device)
        gold_labels = gold_labels.float().to(device)

        # DCG -> discount * CG
        # CG -> sum up each true positive under different label classes
        indices = torch.argsort(torch.argsort(predicted_scores,
                                              descending=True))
        for k in self._k:
            coefficient = self._discounts[indices].to(device)
            coefficient[indices + 1 > k] = 0.0
            batch_dcg = coefficient * gold_labels * mask
            self._dcg[k] += batch_dcg.sum().to(torch.float)
        self._total_num += mask.sum().to(torch.float)

    @overrides
    def reset(self) -> None:
        self._discounts = None
        self._dcg = None
        self._total_num = None

    @overrides
    def get_metric(
            self, reset: bool = False
    ):
        """
        # Returns

        DCG@k : float

        !!! Note
            k in DCG@k is just a place holder.
        """
        metric = {
            f'DCG@{k}': prf_divide(self._dcg[k], self._total_num).item()
            for k in self._k
        }
        if reset:
            self.reset()
        return metric
