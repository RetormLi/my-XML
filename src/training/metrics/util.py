import numpy as np
import torch


def prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result


def select_k_best(
        predicted_scores: torch.Tensor,
        k: int,
        cast_as_indicator: bool = False
) -> torch.Tensor:
    """
    # Parameters

    predicted_scores : `torch.Tensor`, required
        A tensor of predicted scores of shape (batch_size, ...,
        num_label_classes)
    k : `int`, required
        The number of best scores kept.
    cast_as_indicator : `bool`, optional (default = False)
        Return multi-label indicator matrix instead of matrix of scores.
    """
    results = torch.clone(predicted_scores)
    for index in range(results.size(0)):
        # get the row slice
        row = results[index, :]
        # only take the k last elements in the sorted indices,
        # and set them to zero
        row[np.argpartition(row, kth=-k)[:-k]] = 0
        results[index, :] = row
    if cast_as_indicator:
        results = results.bool().to(predicted_scores.dtype)
    return results


def logits_to_predictions(
        logits: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    return torch.sigmoid(logits) > threshold
