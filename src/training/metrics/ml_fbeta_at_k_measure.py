import numpy as np
from overrides import overrides
import torch
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Metric
from src.training.metrics.ml_fbeta_measure import MultiLabelFBetaMeasure
from src.training.metrics.util import select_k_best


@Metric.register('multi_label_fbeta_at_k')
class MultiLabelFBetaAtKMeasure(Metric):
    def __init__(
            self,
            k: Union[int, List[int]],
            beta: float = 1.0,
            labels: List[int] = None
    ) -> None:
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
        # average_options = {None, 'micro', 'macro', 'weighted'}
        # if average not in average_options:
        #     raise ConfigurationError(
        #         f'`average` has to be one of {average_options}.'
        #     )
        if labels is not None and len(labels) == 0:
            raise ConfigurationError(
                '`labels` cannot be an empty list.'
            )
        self._k = k
        self._beta = beta
        self._labels = labels

        self._at_ks: Dict[int, MultiLabelFBetaMeasure] = {
            item: MultiLabelFBetaMeasure(
                beta=beta, average=None, labels=labels
            ) for item in k
        }

    @overrides
    def __call__(
            self,
            predictions: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ):
        for k, inner in self._at_ks.items():
            best_k = select_k_best(
                predictions, k=k, cast_as_indicator=True
            )
            inner(best_k, gold_labels, mask)

    @overrides
    def reset(self) -> None:
        for inner in self._at_ks.values():
            inner.reset()

    @overrides
    def get_metric(
        self, reset: bool = False
    ):
        result = {}
        for k, fbeta in self._at_ks.items():
            inner = fbeta.get_metric(reset)
            result[f'precision@{k}'] = self._get_precision_at_k_from(
                fbeta.true_positive_sum, k=k
            )
            result[f'recall@{k}'] = self._get_recall_at_k_from(
                fbeta.true_positive_sum, fbeta.true_sum
            )
            result[f'fscore@{k}'] = self._get_fscores_at_k_from(
                fbeta.true_positive_sum,
                fbeta.true_sum,
                fbeta.pred_sum,
                beta=self._beta
            )
        if reset:
            self.reset()
        return result

    @staticmethod
    def _get_precision_at_k_from(true_positives, k):
        return np.mean([
            item / k for item in true_positives
        ])

    @staticmethod
    def _get_recall_at_k_from(true_positives, true_sums):
        return np.mean([
            numerator / denominator if denominator != 0 else 0.0
            for numerator, denominator in zip(true_positives, true_sums)
        ])

    @staticmethod
    def _get_fscores_at_k_from(
            true_positives, true_sums, pred_sums,
            beta: float = 1.0):
        return np.mean([
            (1 + beta**2) / (beta**2 * true_sum + pred_sum) * true_positive
            for true_sum, pred_sum, true_positive in zip(
                true_sums, pred_sums, true_positives
            )
        ])
