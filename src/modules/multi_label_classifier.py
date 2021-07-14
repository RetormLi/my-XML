"""
A simple multi-label classifier based on feed-forward neural networks.
"""
from typing import Sequence, Union

import torch
from allennlp.common import Params
from allennlp.modules import FeedForward
from allennlp.nn import Activation


class MultiLabelClassifier(torch.nn.Module):
    """
    This ``Module`` is a simple multi-label classifier, based on feed-forward neural networks.

    Parameters
    ----------
    num_classes : ``int``
        The number of classes.
    input_dim : ``int``
        The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
    num_layers : ``int``
        The number of ``Linear`` layers to apply to the input.
    hidden_dims : ``Union[int, Sequence[int]]``
        The output dimension of each of the ``Linear`` layers.  If this is a single ``int``, we use
        it for all ``Linear`` layers.  If it is a ``Sequence[int]``, ``len(hidden_dims)`` must be
        ``num_layers``.
    activations : ``Union[Callable, Sequence[Callable]]``
        The activation function to use after each ``Linear`` layer.  If this is a single function,
        we use it after all ``Linear`` layers.  If it is a ``Sequence[Callable]``,
        ``len(activations)`` must be ``num_layers``.
    dropout : ``Union[float, Sequence[float]]``, optional
        If given, we will apply this amount of dropout after each layer.  Semantics of ``float``
        versus ``Sequence[float]`` is the same as with other parameters.
    """
    def __init__(self,
                 num_classes: int,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, Sequence[int]],
                 activations: Union[Activation, Sequence[Activation]],
                 share_input: bool = True,
                 dropout: Union[float, Sequence[float]] = 0.0) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._share_input = share_input
        kwargs = {"input_dim": input_dim,
                  "num_layers": num_layers,
                  "hidden_dims": hidden_dims,
                  "activations": activations,
                  "dropout": dropout}
        self._classifiers = torch.nn.ModuleList([
            FeedForward(**kwargs)
            for _ in range(num_classes)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._share_input:
            # list (batch_size) of tensors with shape of [num_class, 2]
            # -> (batch_size, num_class, 2)
            outputs = torch.stack([classifier(inputs) for classifier in self._classifiers],
                                  dim=1)  # stack on dim:`yes_or_no`
        else:
            num_class = inputs.size(1)
            if num_class != self._num_classes:
                raise ValueError(f"Excepted `inputs` with shape of (batch_size, num_class({self._num_classes}), ...), "
                                 f"but got {tuple(inputs.shape)}")
            # Shape: (batch_size, num_class, ...) -> (num_class, batch_size, ...)
            inputs = inputs.transpose(0, 1)
            outputs = torch.stack([classifier(inputs[index])
                                   for index, classifier in enumerate(self._classifiers)],
                                  dim=1)  # stack on dim:`yes_or_no`
        return outputs

    @classmethod
    def from_params(cls, params: Params):
        num_classes = params.pop_int("num_classes")
        input_dim = params.pop_int('input_dim')
        num_layers = params.pop_int('num_layers')
        hidden_dims = params.pop('hidden_dims')
        activations = params.pop('activations')
        share_input = params.pop('share_input', True)
        dropout = params.pop('dropout', 0.0)
        if isinstance(activations, list):
            activations = [Activation.by_name(name)() for name in activations]
        else:
            activations = Activation.by_name(activations)()
        params.assert_empty(cls.__name__)
        return cls(num_classes,
                   input_dim=input_dim,
                   num_layers=num_layers,
                   hidden_dims=hidden_dims,
                   activations=activations,
                   share_input=share_input,
                   dropout=dropout)
