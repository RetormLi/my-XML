from typing import Dict

from overrides import overrides
import torch
from typing import List
from typing import Optional
from typing import Union

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric
from src.training.metrics.util import prf_divide


@Metric.register('nDCG@K')
class NormalizedDiscountedCumulativeGainAtKMeasure(Metric):
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
        # Shape: (num_label_classes, )
        self._normalizer: Union[None, torch.Tensor] = None
        # ndcg -> normalized Discounted Cumulative Gain
        # int -> Tensor (Shape: ())
        self._ndcg: Union[None, Dict[int, torch.Tensor]] = None
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
        """
        # Parameters
        predicted_scores : `torch.Tensor`, required
            A tensor of predicted scores of shape (batch_size,
            num_label_classes).
        gold_labels : `torch.Tensor`, required
            A tensor of integer class label of shape (batch_size,
            num_label_classes). It must be the same shape as the
            `predictions` tensor.
        mask : `torch.Tensor`, optional (default = None).
            A masking tensor, whose shape must be (batch_size, ).
        """
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
            self._normalizer = torch.cumsum(self._discounts, -1).to(device)
            self._ndcg = {
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
            numerator = self._discounts[indices].to(device)
            numerator[indices + 1 > k] = 0.0
            numerator = numerator * gold_labels * mask
            # Shape: (batch_size, )
            numerator = numerator.sum(dim=-1)
            # Shape: (batch_size, )
            denominator = gold_labels.sum(dim=-1).to(device)
            denominator[denominator > k] = k
            zero_indices = denominator == 0
            denominator[zero_indices] = 1
            denominator = denominator - 1
            denominator = self._normalizer[denominator.long()]
            # denominator[zero_indices] = 1.0
            batch_ndcg = numerator / denominator
            self._ndcg[k] += batch_ndcg.sum().to(torch.float)
        self._total_num += mask.sum().to(torch.float)

    @overrides
    def reset(self) -> None:
        self._discounts = None
        self._normalizer = None
        self._ndcg = None
        self._total_num = None

    @overrides
    def get_metric(
        self, reset: bool = False
    ):
        """
        # Returns
        nDCG@k : float
        !!! Note
            k in nDCG@k is just a place holder.
        """
        metric = {
            f'nDCG@{k}': prf_divide(self._ndcg[k], self._total_num).item()
            for k in self._k
        }
        if reset:
            self.reset()
        return metric
