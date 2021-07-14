from allennlp.common.util import ensure_list
from allennlp.data.token_indexers import SingleIdTokenIndexer

from src.common.testing import TestCase
from src.common.testing import util
from src.data.dataset_readers import EURLexReader


class TestMultiLabelDatasetReader(TestCase):
    def setUp(self):
        super().setUp()
        self.sample_dir = self.FIXTURES_ROOT / 'EURLEX'

    def test_read_from_file_with_label_dict(self):
        reader = EURLexReader(
            label_dict=self.sample_dir / 'sample.label_dict.json',
            texts='sample.texts.txt',
            labels='sample.labels.txt',
            token_indexers={'tokens': SingleIdTokenIndexer()}
        )

        instances = ensure_list(
            reader.read(self.sample_dir)
        )

        assert len(instances) == 3
        assert util.text_field_to_text(
            instances[0].fields['text']
        ) == 'A : this is a text'
        # fishery_product processed_foodstuff ship's_flag third_country
        # originating_product export_refund
        assert instances[0].fields['labels'].labels == [1, 2, 3, 4, 5, 6]

        assert util.text_field_to_text(
            instances[1].fields['text']
        ) == 'B : this is a text'
        # beef market_support france award_of_contract
        # aid_to_disadvantaged_groups
        assert instances[1].fields['labels'].labels == [7, 8, 9, 10, 11]

        assert util.text_field_to_text(
            instances[2].fields['text']
        ) == 'C : this is a text'
        # international_agreement international_market shipbuilding
        # south_korea
        assert instances[2].fields['labels'].labels == [12, 13, 14, 15]

    def test_read_from_file_without_label_dict(self):
        reader = EURLexReader(
            texts='sample.texts.txt',
            labels='sample.labels.txt',
            token_indexers={'tokens': SingleIdTokenIndexer()}
        )

        instances = ensure_list(
            reader.read(self.sample_dir)
        )

        assert len(instances) == 3
        assert util.text_field_to_text(
            instances[0].fields['text']
        ) == 'A : this is a text'
        assert set(
            instances[0].fields['labels'].labels
        ) == {
                   'fishery_product', 'processed_foodstuff', "ship's_flag",
                   'third_country', 'originating_product', 'export_refund'
               }

        assert util.text_field_to_text(
            instances[1].fields['text']
        ) == 'B : this is a text'
        assert set(
            instances[1].fields['labels'].labels
        ) == {
                   'beef', 'market_support', 'france', 'award_of_contract',
                   'aid_to_disadvantaged_groups'
               }

        assert util.text_field_to_text(
            instances[2].fields['text']
        ) == 'C : this is a text'
        assert set(
            instances[2].fields['labels'].labels
        ) == {
                   'international_agreement', 'international_market',
                   'shipbuilding',
                   'south_korea'
               }
