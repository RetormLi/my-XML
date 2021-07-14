from src.common.testing import ModelTestCase
from src.models import XMLCNN
from src.data.dataset_readers import EURLexReader


class TestXMLCNN(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / 'xml_cnn.json',
            self.FIXTURES_ROOT / 'EURLEX'
        )

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        pass


class TestSampleXMLCNN(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / 'sample.xml_cnn.json',
            self.FIXTURES_ROOT / 'EURLEX' / 'sample' / 'train'
        )

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        pass
