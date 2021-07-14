from overrides import overrides
import torch
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric
from src.training.metrics.util import prf_divide


@Metric.register('precision@k')
class PrecisionAtKMeasure(Metric):
    def __init__(
            self,
            k: Union[int, List[int]]
    ) -> None:
        if not isinstance(k, List):
            k = [k]
        else:
            k = list(k)
        self._k = k

        # precision
        # int -> Tensor (Shape: ())
        self._precision: Union[None, Dict[int, torch.Tensor]] = None
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

        # Calculate precision (instance-wise)
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
            self._total_num = torch.tensor(0.0, device=device)

        if mask is None:
            mask = torch.ones(batch_size).bool()
        mask = mask.unsqueeze(dim=-1).bool().to(device)
        gold_labels = gold_labels.float().to(device)

        indices = torch.argsort(torch.argsort(predicted_scores,
                                              descending=True))
        for k in self._k:
            true_positives = torch.ones_like(gold_labels)
            true_positives[indices + 1 > k] = 0.0
            true_positives = true_positives * gold_labels * mask
            true_positives = true_positives.sum(dim=-1)
            self._precision[k] += (true_positives / k).sum().to(torch.float)
        self._total_num += mask.sum().to(torch.float)

    @overrides
    def reset(self) -> None:
        self._precision = None
        self._total_num = None

    @overrides
    def get_metric(
        self, reset: bool = False
    ):
        metric = {
            f'P@{k}': prf_divide(self._precision[k], self._total_num).item()
            for k in self._k
        }
        if reset:
            self.reset()
        return metric
