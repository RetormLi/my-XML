from collections import OrderedDict
from itertools import chain
from typing import Dict
from typing import Union

from allennlp.common.file_utils import cached_path
from allennlp.data import Tokenizer


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
