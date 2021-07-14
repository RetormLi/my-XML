import torch

from src.common.testing import TestCase
from src.training.metrics.ml_hamming_loss import MultiLabelHammingLoss


class MultiLabelHammingLossTest(TestCase):
    def test_hamming_loss(self):
        loss = MultiLabelHammingLoss()

        predictions = torch.tensor([[0, 1],
                                    [1, 1]])
        targets = torch.tensor([[0, 1],
                                [0, 1]])
        loss(predictions, targets)
        assert loss.get_metric() == 1.0 / 4.0

        mask = torch.tensor([1, 0])
        loss(predictions, targets, mask)
        assert loss.get_metric() == 1.0 / 6.0

        targets[1, 1] = 0
        loss(predictions, targets)
        assert loss.get_metric() == 3.0 / 10.0

        loss.reset()
        loss(predictions, targets)
        assert loss.get_metric() == 2.0 / 4.0