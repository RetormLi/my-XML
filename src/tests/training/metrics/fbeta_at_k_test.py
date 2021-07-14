import torch

from src.common.testing import TestCase
from src.training.metrics import FBetaAtKMeasure


class FBetaAtKMeasureTest(TestCase):
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

    def test_fbeta_at_1(self):
        fbeta_at_1 = FBetaAtKMeasure(k=1)
        fbeta_at_1(self.prediction_scores, self.targets)
        metric = fbeta_at_1.get_metric()
