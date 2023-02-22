import unittest

import torch

from weakvg.masking import (
    get_concepts_mask,
    get_mask,
    get_multimodal_mask,
    get_proposals_mask,
)


class TestUtils(unittest.TestCase):
    def test_get_proposals_mask(self):
        proposals = torch.tensor([[10, 0, 20, 20], [0, 0, 0, 0]])

        m = get_proposals_mask(proposals)

        self.assertListEqual(m.tolist(), [True, False])

    def test_get_concepts_mask(self):
        heads = torch.tensor([[[227, 15, 0], [0, 0, 0]]])  # [b, q, h] = [1, 2, 3]
        labels = torch.tensor([[3, 7, 0, 4]])  # [b, p] = [1, 4]

        m = get_concepts_mask(heads, labels)

        self.assertEqual(m.shape, (1, 2, 1, 4))

        self.assertListEqual(m[0, 0, 0].tolist(), [True, True, False, True])
        self.assertListEqual(m[0, 1, 0].tolist(), [False, False, False, False])

    def test_get_multimodal_mask(self):
        queries = torch.tensor([[[227, 15, 0], [0, 0, 0]]])  # [b, q, w] = [1, 2, 3]
        proposals = torch.tensor(
            [[[10, 0, 20, 20], [15, 15, 30, 30], [0, 0, 0, 0]]]
        )  # [b, p, 4] = [1, 3, 4]

        m = get_multimodal_mask(queries, proposals)

        self.assertEqual(m.shape, (1, 2, 1, 3))
        self.assertListEqual(m[0, 0, 0].tolist(), [True, True, False])
        self.assertListEqual(m[0, 1, 0].tolist(), [False, False, False])

    def test_get_mask(self):
        queries = torch.tensor([[[227, 15, 0], [0, 0, 0]]])  # [b, q, w] = [1, 2, 3]
        proposals = torch.tensor(
            [[[10, 0, 20, 20], [15, 15, 30, 30], [0, 0, 0, 0]]]
        )  # [b, p, 4] = [1, 3, 4]
        heads = torch.tensor([[[227, 15, 0], [0, 0, 0]]])  # [b, q, h] = [1, 2, 3]
        labels = torch.tensor([[3, 0, 7]])  # [b, p] = [1, 3]

        m = get_mask(queries, proposals, heads, labels)

        expected = get_multimodal_mask(queries, proposals) & get_concepts_mask(
            heads, labels
        )

        self.assertEqual(m.shape, (1, 2, 1, 3))

        self.assertListEqual(m.tolist(), expected.tolist())
