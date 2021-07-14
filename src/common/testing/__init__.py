import pathlib

from allennlp.common import testing
from allennlp.common.testing import AllenNlpTestCase

__all__ = ['TestCase', 'ModelTestCase']


class TestCase(AllenNlpTestCase):
    PROJECT_ROOT = (
            pathlib.Path(__file__).parent / '..' / '..' / '..'
    ).resolve()
    MODULE_ROOT = PROJECT_ROOT / 'src'
    TOOLS_ROOT = MODULE_ROOT / 'tools'
    TESTS_ROOT = MODULE_ROOT / 'tests'
    FIXTURES_ROOT = TESTS_ROOT / 'fixtures'


class ModelTestCase(TestCase, testing.ModelTestCase):
    pass
