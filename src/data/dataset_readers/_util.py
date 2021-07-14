from collections import OrderedDict
from itertools import chain
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL
from allennlp.common.util import START_SYMBOL
from allennlp.data import Field
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data import TokenIndexer
from allennlp.data import Tokenizer
from allennlp.data.fields import MetadataField
from allennlp.data.fields import MultiLabelField
from allennlp.data.fields import TextField

MAXIMUM_LENGTH = 1000000


def build_instance(
        text: str,
        text_tokens: List[Token],
        text_token_indexers: Dict[str, TokenIndexer],
        labels: Optional[str],
        label_dict: Optional[Dict],
        label_split_token: str,
        *,
        labels_are_sequence: bool = False,
        order_labels: bool = True,
        label_token_indexers: Dict[str, TokenIndexer] = None
) -> Optional[Instance]:
    # META -->
    meta = {
        'original_text': text,
    }
    if labels is not None:
        meta['original_labels'] = labels
    meta = MetadataField(meta)

    # TEXT Part  --->
    text_field = TextField(text_tokens,
                           token_indexers=text_token_indexers)

    # LABEL Part --->
    if labels is not None:
        if labels_are_sequence:
            label_tokens = labels.strip().split(label_split_token)
            if label_dict is not None and order_labels:
                label_tokens = sorted(label_tokens,
                                      key=lambda x: -label_dict[x])
            label_tokens.insert(0, START_SYMBOL)
            label_tokens.append(END_SYMBOL)
            label_tokens = [Token(item) for item in label_tokens]
            label_field = TextField(label_tokens,
                                    token_indexers=label_token_indexers)
        else:
            if label_dict is not None:
                label_tokens = [
                    label_dict[item]
                    for item in labels.strip().split(label_split_token)
                    if item in label_dict
                ]
                label_field = MultiLabelField(
                    label_tokens,
                    skip_indexing=True,
                    num_labels=len(label_dict)
                )
            else:
                label_tokens = labels.strip().split(label_split_token)
                label_field = MultiLabelField(label_tokens)

        # training mode, but with an empty label_tokens
        if len(label_tokens) == 0:
            return None
    else:
        # evaluation mode, without `label_field`
        label_field = None

    fields: Dict[str, Field] = {
        'text': text_field,
        'meta': meta
    }
    if label_field is not None:
        fields['labels'] = label_field
    return Instance(fields)


def _label_vocab(path: str) -> Dict[int, str]:
    label_dict = {}
    with open(cached_path(path)) as fh:
        for index, label in enumerate(fh):
            label_dict[index] = label.strip()
    return label_dict


def _ordered_label_dict(label_dict: Dict[int, str], lowercase: bool):
    index2label = OrderedDict()
    label2index = OrderedDict()
    for index in range(len(label_dict)):
        label = label_dict[index]
        index2label[index] = label
        if label in label2index:
            raise RuntimeError(f"`{label}`[line{index+1}] is same as the label [line{label2index[label]+1}]. "
                               f"Please deduplicate first.")
        if lowercase and label.lower() in label2index:
            raise RuntimeError(f"`{label}`[line{index+1}] is same as "
                               f"the label [line{label2index[label.lower()]+1}] after lowercase. "
                               f"Please deduplicate first.")
        if lowercase:
            label = label.lower()
        label2index[label] = index
    return index2label, label2index


class LabelDict:
    def __init__(self,
                 label_vocab: str,
                 replace_vocab: str = None,
                 lowercase: bool = True):
        super().__init__()
        index2label, label2index = _ordered_label_dict(_label_vocab(label_vocab),
                                                       lowercase)
        self._index2label = index2label
        self._label2index = label2index

        if replace_vocab is not None:
            index2replace, replace2index = _ordered_label_dict(_label_vocab(replace_vocab),
                                                               lowercase)
            if len(replace2index) != len(label2index):
                raise ValueError("Except `len(replace2index) == len(label2index)`, "
                                 f"but got `{len(replace2index)} != {len(label2index)}`")
        else:
            index2replace, replace2index = None, None

        self._index2replace = index2replace
        self._replace2index = replace2index

        self._real_labels = None

    def __len__(self):
        return len(self._index2label)

    def __getitem__(self, item: Union[int, str]) -> Union[str, int]:
        if isinstance(item, int):
            if self._replace2index is None:
                return self._index2label[item]
            else:
                label = self._index2label[item]
                replace = self._index2replace[item]
                return f"{label}(replace by {replace})"
        elif isinstance(item, str):
            try:
                return self._label2index[item]
            except KeyError:
                if self._replace2index is not None:
                    return self._replace2index[item]
                else:
                    raise
        else:
            raise ValueError(f'Excepted type: Union[int, str], got: {type(item)}')

    def __contains__(self, item: Union[int, str]) -> bool:
        if isinstance(item, int):
            return item in self._index2label
        elif isinstance(item, str):
            result = item in self._label2index
            if not result and self._replace2index is not None:
                result = item in self._replace2index
            return result
        else:
            raise ValueError(f'Excepted type: Union[int, str], got: {type(item)}')

    def __iter__(self):
        if self._real_labels is None:
            raise RuntimeError("Before call `__iter__`, "
                               "please call `get_tokens` at first.")
        return self._real_labels.__iter__()

    def get_tokens(self, tokenizer: Tokenizer):
        # sometimes, we change labels due to filter stopwords or something
        # like this, so we store the `real_label`
        if self._replace2index is None:
            tokens_list = [tokenizer.tokenize(label)
                           for label in self._label2index]
        else:
            tokens_list = [tokenizer.tokenize(replaced)
                           for replaced in self._replace2index]
        self._real_labels = [
            "".join([token.text for token in tokens])
            for tokens in tokens_list
        ]
        return list(chain(*tokens_list))