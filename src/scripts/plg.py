import click
import numpy as np
from sklearn.utils import check_array
from scipy import sparse


def select_k_best(scores, k_best, cast_as_indicator=False):
    scores = check_array(scores, accept_sparse='csr')

    best_scores = sparse.csr_matrix(scores, copy=True)
    for index in range(best_scores.shape[0]):
        # get the row slice per reference
        row_array = best_scores.data[
            best_scores.indptr[index]: best_scores.indptr[index+1]
        ]
        # only take the k last elements in the sorted indices,
        # and set them to zero
        row_array[np.argpartition(row_array, kth=-k_best)[:-k_best]] = 0

    best_scores.eliminate_zeros()
    if cast_as_indicator:
        best_scores = best_scores.astype(bool).astype(int)
        if sparse.issparse(scores):
            return best_scores
        return best_scores.todense()
    if sparse.issparse(scores):
        return best_scores
    return best_scores.todense()


def test_select_k_best():
    class_probas = np.array([
        [0, 0.9, 0.9, 0.9],
        [0, 0.95, 0.9, 0.95],
        [0.1, 0.2, 0.2, 0.1]
    ])

    # K=1
    # test boolean output
    # when multiple equal highest scores exist, take one of them
    expected_best_1 = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    # sparse
    best_1 = select_k_best(
        sparse.csr_matrix(class_probas),
        k_best=1,
        cast_as_indicator=True
    )
    assert sparse.issparse(best_1)
    assert best_1.todense().tolist() == expected_best_1.tolist()
    # dense
    best_1 = select_k_best(class_probas, k_best=1, cast_as_indicator=True)
    assert not sparse.issparse(best_1)
    assert best_1.tolist() == expected_best_1.tolist()

    # test score output
    expected_best_1_scores = np.array([
        [0, 0, 0, 0.9],
        [0, 0, 0, 0.95],
        [0, 0, 0.2, 0]
    ])
    # sparse
    best_1_scores = select_k_best(
        sparse.csr_matrix(class_probas),
        k_best=1,
        cast_as_indicator=False
    )
    assert sparse.issparse(best_1_scores)
    assert best_1_scores.todense().tolist() == expected_best_1_scores.tolist()
    # dense
    best_1_scores = select_k_best(class_probas, k_best=1,
                                  cast_as_indicator=False)
    assert not sparse.issparse(best_1_scores)
    assert best_1_scores.tolist() == expected_best_1_scores.tolist()

    # K=2
    # test boolean output
    expected_best_2 = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    # sparse
    best_2 = select_k_best(
        sparse.csr_matrix(class_probas),
        k_best=2, cast_as_indicator=True
    )
    assert sparse.issparse(best_2)
    assert best_2.todense().tolist() == expected_best_2.tolist()

    # test score output
    expected_best_2_scores = np.array([
        [0, 0, 0.9, 0.9],
        [0, 0.95, 0, 0.95],
        [0, 0.2, 0.2, 0]
    ])
    # sparse
    best_2_scores = select_k_best(
        sparse.csr_matrix(class_probas),
        k_best=2, cast_as_indicator=False
    )
    assert sparse.issparse(best_2_scores)
    assert best_2_scores.todense().tolist() == expected_best_2_scores.tolist()
    # dense
    best_2_scores = select_k_best(class_probas,
                                  k_best=2, cast_as_indicator=False)
    assert not sparse.issparse(best_2_scores)
    assert best_2_scores.tolist() == expected_best_2_scores.tolist()

    scores = np.array([
        [1, 5, 2, 7, 0],
        [8, 2, 1, 6, 1]
    ])
    # score output
    best_scores = select_k_best(
        sparse.csr_matrix(scores),
        k_best=1, cast_as_indicator=False
    )
    expected = np.array([
        [0, 0, 0, 7, 0],
        [8, 0, 0, 0, 0]
    ])
    assert best_scores.todense().tolist() == expected.tolist()

    # boolean output
    best_scores = select_k_best(
        sparse.csr_matrix(scores),
        k_best=1, cast_as_indicator=True
    )
    expected = np.array([
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0]
    ])
    assert best_scores.todense().tolist() == expected.tolist()


@click.command('Test Range')
@click.argument('text', type=click.FLOAT)
def test_range(text):
    print('hello')


if __name__ == '__main__':
    test_range(0.5)
    test_range('1.5')
