import numpy as np
import torch
from torch.testing import assert_allclose

from src.common.testing import TestCase
from src.training.metrics import MultiLabelFBetaAtKMeasure


class MultiLabelFBetaAtKMeasureTest(TestCase):
    @staticmethod
    def get_fscores_from(true_positives, true_sums, pred_sums, beta=1):
        return np.mean([
            (1 + beta**2) / (beta**2 * true_sum + pred_sum) * true_positive
            for true_sum, pred_sum, true_positive in zip(
                true_sums, pred_sums, true_positives
            )
        ])

    @staticmethod
    def get_precision_at_k_from(true_positives, k):
        return np.mean(
            [item / k for item in true_positives]
        )

    @staticmethod
    def get_recall_at_k_from(true_positives, true_sums):
        return np.mean([
            numerator / denominator if denominator != 0 else 0.0
            for numerator, denominator in zip(true_positives, true_sums)
        ])

    def setUp(self):
        super().setUp()
        self.prediction_scores = torch.tensor(
            [[0.6, 0.3, 0.4, 0.9, 0.8],
             [0.55, 0.65, 0.1, 0.4, 0.25],
             [0.15, 0.75, 0.25, 0.225, 0.2],
             [0.1, 0.45, 0.3, 0.2, 0.4],
             [0.35, 0.4, 0.3, 0.3, 0.55],
             [0.9, 0.8, 0.2, 0.1, 0.8]]
        )
        self.targets = torch.tensor(
            [[1, 0, 1, 0, 0],
             [0, 0, 0, 1, 1],
             [0, 1, 0, 1, 0],
             [1, 0, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [1, 0, 0, 0, 1]]
        )
        self.best_1 = torch.tensor(
            [[0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0]],
            dtype=torch.float
        )
        self.best_1_pred_sum = [1, 3, 0, 1, 1]
        self.best_1_true_sum = [3, 1, 3, 3, 2]
        self.best_1_true_positive_sum = [1, 1, 0, 0, 0]
        self.best_1_true_negative_sum = [3, 3, 3, 2, 3]
        self.best_1_total_sum = [6, 6, 6, 6, 6]
        self.desired_precisions_at_1 = self.get_precision_at_k_from(
            self.best_1_true_positive_sum, k=1
        )
        self.desired_recalls_at_1 = self.get_recall_at_k_from(
            self.best_1_true_positive_sum, self.best_1_true_sum
        )
        self.desired_fscores_at_1 = self.get_fscores_from(
            true_positives=self.best_1_true_positive_sum,
            true_sums=self.best_1_true_sum,
            pred_sums=self.best_1_pred_sum,
            beta=1
        )

        self.best_2 = torch.tensor(
            [[0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1],
             [0, 1, 0, 0, 1],
             [1, 1, 0, 0, 0]],
            dtype=torch.float
        )
        self.best_2_pred_sum = [2, 5, 1, 1, 3]
        self.best_2_true_sum = [3, 1, 3, 3, 2]
        self.best_2_true_positive_sum = [1, 1, 0, 0, 0]
        self.best_2_true_negative_sum = [2, 1, 2, 2, 1]
        self.best_2_total_sum = [6, 6, 6, 6, 6]
        self.desired_precisions_at_2 = self.get_precision_at_k_from(
            self.best_2_true_positive_sum, k=2
        )
        self.desired_recalls_at_2 = self.get_recall_at_k_from(
            self.best_2_true_positive_sum, self.best_2_true_sum
        )
        self.desired_fscores_at_2 = self.get_fscores_from(
            true_positives=self.best_2_true_positive_sum,
            true_sums=self.best_2_true_sum,
            pred_sums=self.best_2_pred_sum,
            beta=1
        )

        self.best_3 = torch.tensor(
            [[1, 0, 0, 1, 1],
             [1, 1, 0, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 0, 1],
             [1, 1, 0, 0, 1],
             [1, 1, 0, 0, 1]],
            dtype=torch.float
        )
        self.best_3_pred_sum = [4, 5, 2, 3, 4]
        self.best_3_true_sum = [3, 1, 3, 3, 2]
        self.best_3_true_positive_sum = [2, 1, 1, 2, 1]
        self.best_3_true_negative_sum = [1, 1, 2, 2, 1]
        self.best_3_total_sum = [6, 6, 6, 6, 6]
        self.desired_precisions_at_3 = self.get_precision_at_k_from(
            self.best_3_true_positive_sum, k=3
        )
        self.desired_recalls_at_3 = self.get_recall_at_k_from(
            self.best_3_true_positive_sum, self.best_3_true_sum
        )
        self.desired_fscores_at_3 = self.get_fscores_from(
            true_positives=self.best_3_true_positive_sum,
            true_sums=self.best_3_true_sum,
            pred_sums=self.best_3_pred_sum,
            beta=1
        )

    def test_multilabel_fbeta_at_1_state(self):
        ml_fbeta_at_1 = MultiLabelFBetaAtKMeasure(k=1)
        ml_fbeta_at_1(self.prediction_scores, self.targets)

        assert ml_fbeta_at_1._k == [1]
        assert list(ml_fbeta_at_1._at_ks.keys()) == [1]

        inner_at_1 = ml_fbeta_at_1._at_ks[1]
        assert_allclose(inner_at_1._pred_sum.tolist(), self.best_1_pred_sum)
        assert_allclose(inner_at_1._true_sum.tolist(), self.best_1_true_sum)
        assert_allclose(
            inner_at_1._true_positive_sum.tolist(),
            self.best_1_true_positive_sum
        )
        assert_allclose(
            inner_at_1._true_negative_sum.tolist(),
            self.best_1_true_negative_sum
        )
        assert_allclose(inner_at_1._total_sum.tolist(), self.best_1_total_sum)

    def test_multilabel_fbeta_at_1_metric(self):
        ml_fbeta_at_1 = MultiLabelFBetaAtKMeasure(k=1)
        ml_fbeta_at_1(self.prediction_scores, self.targets)

        metric = ml_fbeta_at_1.get_metric()
        assert len(metric) == 3

        precisions_at_1 = metric['precision@1']
        recalls_at_1 = metric['recall@1']
        fscores_at_1 = metric['fscore@1']

        assert_allclose(precisions_at_1, self.desired_precisions_at_1)
        assert_allclose(recalls_at_1, self.desired_recalls_at_1)
        assert_allclose(fscores_at_1, self.desired_fscores_at_1)

    def test_multilabel_fbeta_at_1_2_3(self):
        ml_fbeta_at_1_2_3 = MultiLabelFBetaAtKMeasure(k=[1, 2, 3])
        ml_fbeta_at_1_2_3(self.prediction_scores, self.targets)
        # check state
        assert ml_fbeta_at_1_2_3._k == [1, 2, 3]
        assert list(ml_fbeta_at_1_2_3._at_ks.keys()) == [1, 2, 3]

        inner_at_1 = ml_fbeta_at_1_2_3._at_ks[1]
        assert_allclose(inner_at_1._pred_sum.tolist(), self.best_1_pred_sum)
        assert_allclose(inner_at_1._true_sum.tolist(), self.best_1_true_sum)
        assert_allclose(
            inner_at_1._true_positive_sum.tolist(),
            self.best_1_true_positive_sum
        )
        assert_allclose(
            inner_at_1._true_negative_sum.tolist(),
            self.best_1_true_negative_sum
        )
        assert_allclose(inner_at_1._total_sum.tolist(), self.best_1_total_sum)

        inner_at_2 = ml_fbeta_at_1_2_3._at_ks[2]
        assert_allclose(inner_at_2._pred_sum.tolist(), self.best_2_pred_sum)
        assert_allclose(inner_at_2._true_sum.tolist(), self.best_2_true_sum)
        assert_allclose(
            inner_at_2._true_positive_sum.tolist(),
            self.best_2_true_positive_sum
        )
        assert_allclose(
            inner_at_2._true_negative_sum.tolist(),
            self.best_2_true_negative_sum
        )
        assert_allclose(inner_at_2._total_sum.tolist(), self.best_2_total_sum)

        inner_at_3 = ml_fbeta_at_1_2_3._at_ks[3]
        assert_allclose(inner_at_3._pred_sum.tolist(), self.best_3_pred_sum)
        assert_allclose(inner_at_3._true_sum.tolist(), self.best_3_true_sum)
        assert_allclose(
            inner_at_3._true_positive_sum.tolist(),
            self.best_3_true_positive_sum
        )
        assert_allclose(
            inner_at_3._true_negative_sum.tolist(),
            self.best_3_true_negative_sum
        )
        assert_allclose(inner_at_3._total_sum.tolist(), self.best_3_total_sum)

        # check value
        metric = ml_fbeta_at_1_2_3.get_metric()
        assert len(metric) == 3 * 3

        precisions_at_1 = metric['precision@1']
        recalls_at_1 = metric['recall@1']
        fscores_at_1 = metric['fscore@1']
        precisions_at_2 = metric['precision@2']
        recalls_at_2 = metric['recall@2']
        fscores_at_2 = metric['fscore@2']
        precisions_at_3 = metric['precision@3']
        recalls_at_3 = metric['recall@3']
        fscores_at_3 = metric['fscore@3']

        assert_allclose(precisions_at_1, self.desired_precisions_at_1)
        assert_allclose(recalls_at_1, self.desired_recalls_at_1)
        assert_allclose(fscores_at_1, self.desired_fscores_at_1)
        assert_allclose(precisions_at_2, self.desired_precisions_at_2)
        assert_allclose(recalls_at_2, self.desired_recalls_at_2)
        assert_allclose(fscores_at_2, self.desired_fscores_at_2)
        assert_allclose(precisions_at_3, self.desired_precisions_at_3)
        assert_allclose(recalls_at_3, self.desired_recalls_at_3)
        assert_allclose(fscores_at_3, self.desired_fscores_at_3)
