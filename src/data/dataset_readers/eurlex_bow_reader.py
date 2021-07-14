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
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Tokenizer
from allennlp.data.fields import MetadataField
from allennlp.data.fields import MultiLabelField
from allennlp.data.fields import TextField
from allennlp.data.fields import ArrayField
from allennlp.data.tokenizers import WordTokenizer
from overrides import overrides
import numpy as np

logger = logging.getLogger(__name__)

MAXIMUM_LENGTH = 1000000


# my

@DatasetReader.register('eurlex_bow_reader')
class EURLexBowReader(DatasetReader):
    def __init__(self,
                 label_size: int,
                 feature_size: int,
                 max_num_tokens: int = 500,
                 tokenizer: Tokenizer = WordTokenizer(),
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.max_num_tokens = max_num_tokens
        self.tokenizer = tokenizer
        self.lazy = lazy
        self.feature_size = feature_size
        self.label_size = label_size

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info(f'Reading instances from {file_path}...')

        with open(file_path) as fh:
            row_num, vocab_size, label_dim = [int(num) for num in fh.readline().split(' ')]
            for line in fh:
                line = line.strip()
                split_index = line.find(' ')

                if line[:split_index].find(':') == -1:
                    labels, text = line[:split_index], line[split_index + 1:]
                else:
                    return
                    # labels = ''
                    # text = line

                instance = self.text_to_instance(text, labels)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,
                         text: str,
                         label: str = None) -> Optional[Instance]:
        # text: "0:0.1 1:1.1..."
        # label: "3,4,52,260,518,992,1543"
        # META -->
        pairs = dict()
        for pair in text.strip().split():
            split_index = pair.find(':')
            pairs[int(pair[:split_index])] = float(pair[split_index + 1:])

        onehot_tfidf = np.zeros(self.feature_size)
        for word in pairs:
            onehot_tfidf[word] = pairs[word]
        text_field = ArrayField(onehot_tfidf)

        meta = {
            'original_text': text,
            'text_tfidfs': onehot_tfidf
        }
        if label is not None:
            labels = label.strip().split(',')
            onehot_labels = np.zeros(self.label_size).astype(float)
            if len(labels) == 0 or labels == ['']:
                return None
            else:
                labels = [int(i) for i in labels]
                for i in labels:
                    onehot_labels[i] = 1.0
            meta['labels'] = onehot_labels
            label_field = MultiLabelField(labels, skip_indexing=True, num_labels=self.label_size)
        else:
            label_field = None

        meta = MetadataField(meta)

        fields: Dict[str, Field] = {
            'meta': meta,
            'text': text_field,
            # 'labels': label_field
        }
        if label is not None:
            fields['label'] = label_field
        return Instance(fields)


if __name__ == '__main__':
    test = EURLexBowReader(3993, 5000)
    instances = test.read("E:\\NLP\\XML-Reasoner\\data\\Eurlex4kbow\\eurlex_little.txt")
    print(instances)
