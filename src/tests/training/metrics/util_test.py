import torch
from torch.testing import assert_allclose

from src.common.testing import TestCase
from src.training.metrics.util import select_k_best


class UtilTest(TestCase):
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
        self.best_1 = torch.tensor(
            [[0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0]],
            dtype=torch.float
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
        self.best_3 = torch.tensor(
            [[1, 0, 0, 1, 1],
             [1, 1, 0, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 0, 1],
             [1, 1, 0, 0, 1],
             [1, 1, 0, 0, 1]],
            dtype=torch.float
        )

    def test_select_1_best(self):
        best_1 = select_k_best(
            self.prediction_scores,
            k=1, cast_as_indicator=True
        )
        assert_allclose(best_1, self.best_1)

    def test_select_1_best_scores(self):
        best_1_scores = select_k_best(
            self.prediction_scores,
            k=1, cast_as_indicator=False
        )
        expected_best_1_score = torch.tensor(
            [[0, 0, 0, 0.9, 0],
             [0, 0.65, 0, 0, 0],
             [0, 0.75, 0, 0, 0],
             [0, 0.45, 0, 0, 0],
             [0, 0, 0, 0, 0.55],
             [0.9, 0, 0, 0, 0]]
        )
        assert_allclose(best_1_scores, expected_best_1_score)

    def test_select_2_best(self):
        best_2 = select_k_best(
            self.prediction_scores,
            k=2, cast_as_indicator=True
        )
        assert_allclose(best_2, self.best_2)

    def test_select_2_best_scores(self):
        best_2_scores = select_k_best(
            self.prediction_scores,
            k=2, cast_as_indicator=False
        )
        expected_best_2_score = torch.tensor(
            [[0, 0, 0, 0.9, 0.8],
             [0.55, 0.65, 0, 0, 0],
             [0, 0.75, 0.25, 0, 0],
             [0, 0.45, 0, 0, 0.4],
             [0, 0.4, 0, 0, 0.55],
             [0.9, 0.8, 0, 0, 0]]
        )
        assert_allclose(best_2_scores, expected_best_2_score)

    def test_select_3_best_and_3_best_scores(self):
        best_3 = select_k_best(
            self.prediction_scores,
            k=3, cast_as_indicator=True
        )
        assert_allclose(best_3, self.best_3)

        best_3_scores = select_k_best(
            self.prediction_scores,
            k=3, cast_as_indicator=False
        )
        expected_best_3_scores = torch.tensor(
            [[0.6, 0, 0, 0.9, 0.8],
             [0.55, 0.65, 0, 0.4, 0],
             [0, 0.75, 0.25, 0.225, 0],
             [0, 0.45, 0.3, 0, 0.4],
             [0.35, 0.4, 0, 0, 0.55],
             [0.9, 0.8, 0, 0, 0.8]]
        )
        assert_allclose(best_3_scores, expected_best_3_scores)
