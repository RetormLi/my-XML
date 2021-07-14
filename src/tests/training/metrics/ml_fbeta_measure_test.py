import torch
from torch.testing import assert_allclose
from typing import List

from allennlp.common.checks import ConfigurationError
from src.common.testing import TestCase
from src.training.metrics import MultiLabelFBetaMeasure


class MultiLabelFBetaMeasureTest(TestCase):
    def setUp(self):
        super().setUp()
        # [[1, 0, 0, 1, 1],
        #  [1, 1, 0, 0, 0],
        #  [0, 1, 0, 0, 0],
        #  [0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 1],
        #  [1, 1, 0, 0, 1]]
        # -> [3, 3, 0, 1, 3]
        self.predictions = torch.tensor(
            [[1, 0, 0, 1, 1],
             [1, 1, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1],
             [1, 1, 0, 0, 1]]
        )
        self.threshold = 0.5
        self.prediction_scores = torch.tensor(
            [[0.6, 0.3, 0.4, 0.9, 0.8],
             [0.55, 0.65, 0.1, 0.4, 0.25],
             [0.15, 0.75, 0.25, 0.2, 0.2],
             [0.1, 0.4, 0.3, 0.2, 0.4],
             [0.3, 0.4, 0.3, 0.3, 0.55],
             [0.9, 0.8, 0.2, 0.1, 0.8]]
        )
        # -> [3, 1, 3, 3, 2]
        self.targets = torch.tensor(
            [[1, 0, 1, 0, 0],
             [0, 0, 0, 1, 1],
             [0, 1, 0, 1, 0],
             [1, 0, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [1, 0, 0, 0, 1]]
        )

        # detailed target state
        self.pred_sum = [3, 3, 0, 1, 3]
        self.true_sum = [3, 1, 3, 3, 2]
        self.true_positive_sum = [2, 1, 0, 0, 1]
        self.true_negative_sum = [2, 3, 3, 2, 2]
        self.total_sum = [6, 6, 6, 6, 6]

        desired_precisions = [2 / 3, 1 / 3, 0.00, 0.00, 1 / 3]
        desired_recalls = [2 / 3, 1 / 1, 0.00, 0.00, 1 / 2]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]
        self.desired_precisions = desired_precisions
        self.desired_recalls = desired_recalls
        self.desired_fscores = desired_fscores

    def test_cast_as_indicator(self):
        casted = MultiLabelFBetaMeasure.cast_as_indicator(
            self.prediction_scores, self.threshold
        )
        assert_allclose(
            casted, self.predictions
        )

    def test_config_errors(self):
        # Bad beta
        self.assertRaises(ConfigurationError,
                          MultiLabelFBetaMeasure,
                          beta=0.0)

        # Bad average option
        self.assertRaises(ConfigurationError,
                          MultiLabelFBetaMeasure,
                          average='mega')

        # Empty input labels
        self.assertRaises(ConfigurationError,
                          MultiLabelFBetaMeasure,
                          labels=[])

    def test_runtime_errors(self):
        multi_fbeta = MultiLabelFBetaMeasure()
        # Metric was never called.
        self.assertRaises(RuntimeError, multi_fbeta.get_metric)

    def test_multilabel_fbeta_state(self):
        ml_fbeta = MultiLabelFBetaMeasure()
        ml_fbeta(self.predictions, self.targets)

        # check state
        assert_allclose(ml_fbeta._pred_sum.tolist(), self.pred_sum)
        assert_allclose(ml_fbeta._true_sum.tolist(), self.true_sum)
        assert_allclose(ml_fbeta._true_positive_sum.tolist(),
                        self.true_positive_sum)
        assert_allclose(ml_fbeta._true_negative_sum.tolist(),
                        self.true_negative_sum)
        assert_allclose(ml_fbeta._total_sum.tolist(), self.total_sum)

    def test_multilabel_fbeta_metric(self):
        ml_fbeta = MultiLabelFBetaMeasure()
        ml_fbeta(self.predictions, self.targets)

        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']
        fscores = metric['fscore']

        # check value
        assert_allclose(precisions, self.desired_precisions)
        assert_allclose(recalls, self.desired_recalls)
        assert_allclose(fscores, self.desired_fscores)

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_multilabel_fbeta_with_mask(self):
        mask = torch.tensor(
            [True, True, True, True, True, False],
            dtype=torch.bool
        )

        ml_fbeta = MultiLabelFBetaMeasure()
        ml_fbeta(self.predictions, self.targets, mask)

        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']
        fscores = metric['fscore']

        assert_allclose(ml_fbeta._pred_sum.tolist(), [2, 2, 0, 1, 2])
        assert_allclose(ml_fbeta._true_sum.tolist(), [2, 1, 3, 3, 1])
        assert_allclose(ml_fbeta._true_positive_sum.tolist(), [1, 1, 0, 0, 0])

        desired_precisions = [1 / 2, 1 / 2, 0.0, 0.0, 0.0]
        desired_recalls = [1 / 2, 1 / 1, 0.0, 0.0, 0.0]
        desired_fscores = [
            (2 * p * r) / (p + r) if p + r != 0.0 else 0.0
            for p, r in zip(desired_precisions, desired_recalls)
        ]

        assert_allclose(precisions, desired_precisions)
        assert_allclose(recalls, desired_recalls)
        assert_allclose(fscores, desired_fscores)

    def test_multilabel_fbeta_macro_average_metric(self):
        ml_fbeta = MultiLabelFBetaMeasure(average='macro')
        ml_fbeta(self.predictions, self.targets)

        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']
        fscores = metric["fscore"]

        # We keep the expected values in CPU because MultiLabelFBetaMeasure
        # returns them in CPU.
        macro_precision = torch.tensor(self.desired_precisions).mean()
        macro_recall = torch.tensor(self.desired_recalls).mean()
        macro_fscore = torch.tensor(self.desired_fscores).mean()

        # check value
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

        # check type
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    def test_multilabel_fbeta_micro_average_metric(self):
        ml_fbeta = MultiLabelFBetaMeasure(average='micro')
        ml_fbeta(self.predictions, self.targets)

        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']
        fscores = metric["fscore"]

        # We keep the expected values in CPU because MultiLabelFBetaMeasure
        # returns them in CPU.
        true_positive = torch.tensor([2, 1, 0, 0, 1], dtype=torch.float)
        false_positive = torch.tensor([1, 2, 0, 1, 2], dtype=torch.float)
        false_negative = torch.tensor([1, 0, 3, 3, 1], dtype=torch.float)
        mean_true_positive = true_positive.mean()
        mean_false_positive = false_positive.mean()
        mean_false_negative = false_negative.mean()
        micro_precision = mean_true_positive / (
            mean_true_positive + mean_false_positive
        )
        micro_recall = mean_true_positive / (
            mean_true_positive + mean_false_negative
        )
        micro_fscore = (2 * micro_precision * micro_recall) / (
            micro_precision + micro_recall
        )

        # check value
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

        # check type
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    def test_multilabel_fbeta_with_explicit_label_orders(self):
        # same prediction but with an explicit label ordering
        ml_fbeta = MultiLabelFBetaMeasure(labels=[4, 3, 2, 1, 0])
        ml_fbeta(self.predictions, self.targets)

        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']
        fscores = metric['fscore']

        # check value
        assert_allclose(precisions, self.desired_precisions[::-1])
        assert_allclose(recalls, self.desired_recalls[::-1])
        assert_allclose(fscores, self.desired_fscores[::-1])

        # check type
        assert isinstance(precisions, List)
        assert isinstance(recalls, List)
        assert isinstance(fscores, List)

    def test_multilabel_fbeta_with_macro_average_and_explicit_labels(self):
        labels = [0, 1]
        ml_fbeta = MultiLabelFBetaMeasure(average='macro', labels=[0, 1])
        ml_fbeta(self.predictions, self.targets)
        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']
        fscores = metric['fscore']

        # We keep the expected values in CPU because MultiLabelFbetaMeasure
        # returns them in CPU.
        macro_precision = torch.tensor(self.desired_precisions)[labels].mean()
        macro_recall = torch.tensor(self.desired_recalls)[labels].mean()
        macro_fscore = torch.tensor(self.desired_fscores)[labels].mean()

        # check value
        assert_allclose(precisions, macro_precision)
        assert_allclose(recalls, macro_recall)
        assert_allclose(fscores, macro_fscore)

    def test_multilabel_fbeta_with_micro_average_and_explict_labels(self):
        labels = [1, 3]
        ml_fbeta = MultiLabelFBetaMeasure(average='micro', labels=labels)
        ml_fbeta(self.predictions, self.targets)
        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']
        fscores = metric["fscore"]

        # We keep the expected values in CPU because MultiLabelFBetaMeasure
        # returns them in CPU.
        true_positive = torch.tensor([1, 0], dtype=torch.float)
        false_positive = torch.tensor([2, 1], dtype=torch.float)
        false_negative = torch.tensor([0, 3], dtype=torch.float)
        mean_true_positive = true_positive.mean()
        mean_false_positive = false_positive.mean()
        mean_false_negative = false_negative.mean()
        micro_precision = mean_true_positive / (
                mean_true_positive + mean_false_positive
        )
        micro_recall = mean_true_positive / (
                mean_true_positive + mean_false_negative
        )
        micro_fscore = (2 * micro_precision * micro_recall) / (
                micro_precision + micro_recall
        )

        # check value
        assert_allclose(precisions, micro_precision)
        assert_allclose(recalls, micro_recall)
        assert_allclose(fscores, micro_fscore)

        # check type
        assert isinstance(precisions, float)
        assert isinstance(recalls, float)
        assert isinstance(fscores, float)

    def test_multilabel_fbeta_handles_batch_size_of_one(self):
        predictions = torch.tensor([[1, 0, 0, 1, 1]])
        targets = torch.tensor([[1, 0, 1, 0, 0]])
        mask = torch.tensor([True], dtype=torch.bool)

        ml_fbeta = MultiLabelFBetaMeasure()
        ml_fbeta(predictions, targets, mask)
        metric = ml_fbeta.get_metric()
        precisions = metric['precision']
        recalls = metric['recall']

        assert_allclose(precisions, [1.0, 0.0, 0.0, 0.0, 0.0])
        assert_allclose(recalls, [1.0, 0.0, 0.0, 0.0, 0.0])
