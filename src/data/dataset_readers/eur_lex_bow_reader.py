import json
import logging
from pathlib import Path
from typing import Dict, List
from typing import Iterable
from typing import Optional

from allennlp.data import DatasetReader, Field, Vocabulary
from allennlp.data import Instance
from allennlp.data import TokenIndexer
from allennlp.data import Tokenizer
from allennlp.data.fields import MetadataField, TextField, MultiLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)

MAXIMUM_LENGTH = 1000000


@DatasetReader.register('eur_lex_bow_reader')
class EURLexBowReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,    #: TokenIndexer = SingleIdTokenIndexer,
                 max_num_tokens: int = 500,
                 tokenizer: Tokenizer = WordTokenizer(),
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        if token_indexers is None:
            token_indexers = {'tokens': SingleIdTokenIndexer()}

        self.token_indexers = token_indexers
        self.max_num_tokens = max_num_tokens
        self.tokenizer = tokenizer
        self.lazy = lazy

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # 直接传入文件名 file_path: /Users/sdj/Documents/research/MLC/XML-Reasoner.method/data/Eurlex_BOW/eurlex_test.txt
        logger.info("Reading file at %s", file_path)

        with open(file_path) as dataset_file:
            # train15539/dev3809, 5000, 3993
            row_num, vocab_size, label_dim = [int(num) for num in dataset_file.readline().split(' ')]

            data_raw = dataset_file.read().split('\n')[:-1]
            for row in data_raw:
                row = row.strip()
                label = ''
                space_idx = row.find(' ')
                if row[:space_idx].find(':') == -1:  # sample doesnt have label
                    label = row[:space_idx]
                    row = row[space_idx + 1:]
                else:
                    return
                text = row
                instance = self.text_to_instance(text, label)

                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,
                         text: str,
                         label: str = None) -> Optional[Instance]:
        # META -->
        meta = {
            'original_text': text,
            'text_tfidfs': [float(i[i.find(':') + 1:]) for i in text.split(' ')]
        }
        if label is not None:
            meta['original_labels'] = label
        meta = MetadataField(meta)

        # TEXT -->
        text = ' '.join([i[:i.find(':')] for i in text.split(' ')])
        text_tokens = self.tokenizer.tokenize(text[:MAXIMUM_LENGTH])  # [:self.max_num_tokens]
        text_field = TextField(text_tokens, token_indexers=self.token_indexers)

        # LABEL -->
        if label is not None:
            label_tokens = label.strip().split(',')
            label_field = MultiLabelField(label_tokens)
            if len(label_tokens) == 0:
                return None
        else:
            label_field = None

        fields: Dict[str, Field] = {
            'meta': meta,
            'text': text_field,
            'labels': label_field
        }
        return Instance(fields)


if __name__ == '__main__':
    test = EURLexBowReader()
    instances = test.read("E:\\NLP\\XML-Reasoner\\data\\Eurlex4kbow\\eurlex_little.txt")
    print(instances)