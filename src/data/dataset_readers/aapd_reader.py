import json
import logging
from typing import Dict
from typing import Iterable
from typing import Optional

from allennlp.data import DatasetReader
from allennlp.data import Instance
from allennlp.data import TokenIndexer
from allennlp.data import Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

from . import _util

logger = logging.getLogger(__name__)


@DatasetReader.register('aapd_reader')
class AAPDReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 label_dict: str = None,
                 lowercase: bool = True,
                 max_num_tokens: int = 256,
                 split_label_token: str = ' ',
                 tokenizer: Tokenizer = WordTokenizer(),
                 labels_are_sequence: bool = False,
                 order_labels: bool = True,
                 label_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        if label_dict is not None:
            with open(label_dict) as fh:
                self.label_dict = json.load(fh)
        else:
            self.label_dict = None

        self.lowercase = lowercase
        self.max_num_tokens = max_num_tokens
        self.split_label_token = split_label_token
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        #: 是否将标签集合视为序列
        self.labels_are_sequence = labels_are_sequence
        #: 是否对标签进行排序
        #: 只在 labels_are_sequence == True 的情形下才适用
        self.order_labels = order_labels
        #: 将 label 转换为对应的 indexer
        self.label_token_indexers = label_token_indexers
        self.lazy = lazy

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info(f'Reading instances from {file_path}...')

        with open(file_path) as fh:
            for text in fh:
                labels = fh.readline()
                instance = self.text_to_instance(text, labels)
                if instance is None:
                    continue
                else:
                    yield instance

    @overrides
    def text_to_instance(self,
                         text: str,
                         labels: str = None) -> Optional[Instance]:
        return _util.build_instance(
            text,
            self.tokenizer.tokenize(
                text[:_util.MAXIMUM_LENGTH]
            )[:self.max_num_tokens],
            self.token_indexers,
            labels,
            self.label_dict,
            self.split_label_token,
            labels_are_sequence=self.labels_are_sequence,
            order_labels=self.order_labels,
            label_token_indexers=self.label_token_indexers
        )
