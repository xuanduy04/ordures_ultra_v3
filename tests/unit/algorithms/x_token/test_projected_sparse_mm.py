import unittest

import torch

from nemo_rl.algorithms.x_token.loss_utils import Fp32SparseProjectSelected


class TestProjectedSparseMM(unittest.TestCase):
    def test_selected_probabilities_and_gradients_match_full_projection(self):
        rng = torch.Generator().manual_seed(42)
        matrix = torch.tensor(
            [[0.25, 0.0, 0.75, 0.0], [0.0, 0.4, 0.1, 0.5], [0.15, 0.35, 0.0, 0.5]]
        ).to_sparse()
        probabilities = torch.rand(17, 3, generator=rng, requires_grad=True)
        rows = 2 * Fp32SparseProjectSelected.ROW_BLOCK + 7
        groups = torch.randint(17, (rows,), generator=rng)
        ids = torch.randint(4, (rows, 3), generator=rng)
        grad = torch.rand(ids.shape, generator=rng)

        actual = Fp32SparseProjectSelected.apply(matrix, probabilities, groups, ids)
        projected = torch.sparse.mm(matrix.t(), probabilities.t()).t()
        expected = projected[groups[:, None], ids]
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual, probabilities, grad)[0]
        expected_grad = torch.autograd.grad(expected, probabilities, grad)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
