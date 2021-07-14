from src.common.testing import TestCase
from src.data.dataset_readers.util import LabelDict


class TestDatasetUtil(TestCase):
    def setUp(self):
        super().setUp()
        self.label_vocab = self.FIXTURES_ROOT / "experiment.labels.vocab.txt"
        self.replace_vocab = self.FIXTURES_ROOT / "experiment.replaces.vocab.txt"

    def test_label_vocab(self):
        label_dict = LabelDict(label_vocab=str(self.label_vocab),
                               lowercase=False)

        # test __len__
        assert len(label_dict) == 12

        # test __getitem__
        assert label_dict[0] == 'cs.IT'
        assert label_dict[11] == 'cs.CR'
        assert label_dict['cs.IT'] == 0
        assert label_dict['cs.CR'] == 11

        # test __contains___
        assert 0 in label_dict
        assert len(label_dict) not in label_dict
        assert 'cs.IT' in label_dict
        assert 'not_exist' not in label_dict

        # test __iter__
        for item, expected in zip(label_dict, label_dict._label2index):
            assert item == expected

    def test_lowercase_label_vocab(self):
        label_dict = LabelDict(label_vocab=str(self.label_vocab),
                               lowercase=True)

        assert label_dict[0] == 'cs.IT'.lower()

    def test_label_vocab_with_replace_vocab(self):
        label_dict = LabelDict(label_vocab=str(self.label_vocab),
                               replace_vocab=str(self.replace_vocab),
                               lowercase=False)

        # test __len__
        assert len(label_dict) == 12

        # test __getitem__
        assert label_dict[0] == 'cs.IT(replace by [unused0])'
        assert label_dict[11] == 'cs.CR(replace by [unused11])'
        assert label_dict['cs.IT'] == 0
        assert label_dict['[unused0]'] == 0
        assert label_dict['cs.CR'] == 11
        assert label_dict['[unused11]'] == 11

        # test __contains__
        assert 0 in label_dict
        assert -1 not in label_dict
        assert 'cs.IT' in label_dict
        assert '[unused0]' in label_dict
        assert 'not_exist' not in label_dict

        # test __iter__
        for item, expected in zip(label_dict, label_dict._replace2index):
            assert item == expected
