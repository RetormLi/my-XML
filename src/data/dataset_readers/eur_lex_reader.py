import json
import logging
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import Optional

from allennlp.data import DatasetReader
from allennlp.data import Field
from allennlp.data import Instance
from allennlp.data import TokenIndexer
from allennlp.data import Tokenizer
from allennlp.data.fields import MetadataField
from allennlp.data.fields import MultiLabelField
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)

_MAXIMUM_LENGTH = 1000000


@DatasetReader.register('eur_lex_reader')
class EURLexReader(DatasetReader):
    def __init__(self,
                 texts: str,
                 labels: str,
                 token_indexers: Dict[str, TokenIndexer],
                 label_dict: str = None,
                 lowercase: bool = True,
                 max_num_tokens: int = 256,
                 split_label_token: str = ' ',
                 tokenizer: Tokenizer = WordTokenizer(),
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        if label_dict is not None:
            with open(label_dict) as fh:
                self.label_dict = json.load(fh)
        else:
            self.label_dict = None

        self.texts = texts
        self.labels = labels
        self.lowercase = lowercase
        self.max_num_tokens = max_num_tokens
        self.split_label_token = split_label_token
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.lazy = lazy

    @overrides
    def _read(self, dir_path: str) -> Iterable[Instance]:
        texts = Path(dir_path) / self.texts
        labels = Path(dir_path) / self.labels
        logger.info(f'Reading instances from {texts} and {labels}')

        with open(texts) as fh_texts, open(labels) as fh_labels:
            for text, label in zip(fh_texts.readlines(),
                                   fh_labels.readlines()):
                instance = self.text_to_instance(
                    text, label
                )
                if instance is None:
                    continue
                else:
                    yield instance

    @overrides
    def text_to_instance(self,
                         text: str,
                         labels: str = None) -> Optional[Instance]:
        text_tokens = self.tokenizer.tokenize(text[:_MAXIMUM_LENGTH])
        text_field = TextField(text_tokens[:self.max_num_tokens],
                               token_indexers=self.token_indexers)
        if labels is not None:
            if self.label_dict is not None:
                labels = [
                    self.label_dict[item]
                    for item in labels.strip().split(self.split_label_token)
                    if item in self.label_dict
                ]
            else:
                labels = labels.strip().split(self.split_label_token)

            if len(labels) == 0:
                return None

        meta = {
            'original_text': text
        }
        if labels is not None:
            meta['original_labels'] = labels

        fields: Dict[str, Field] = {
            'text': text_field,
            'meta': MetadataField(meta)
        }
        if labels is not None:
            if self.label_dict is not None:
                fields['labels'] = MultiLabelField(
                    labels,
                    skip_indexing=True,
                    num_labels=len(self.label_dict)
                )
            else:
                fields['labels'] = MultiLabelField(
                    labels
                )

        return Instance(fields)
