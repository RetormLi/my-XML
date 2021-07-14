import torch

from src.common.testing import TestCase
from src.training.metrics import DiscountedCumulativeGainAtKMeasure


class DiscountedCumulativeGainAtKMeasureTest(TestCase):
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

    def test_dcg_at_1_and_2(self):
        DCG_at_1_and_2 = DiscountedCumulativeGainAtKMeasure(k=[1, 2])
        DCG_at_1_and_2(self.prediction_scores, self.targets)
        metric = DCG_at_1_and_2.get_metric()
        pass
