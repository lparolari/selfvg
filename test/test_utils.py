import unittest

import torch

from weakvg.utils import tlbr2ctwh, tlbr2tlwh


class TestUtils(unittest.TestCase):
    def test_tlbr2tlwh(self):
        tlbr = torch.tensor([[10, 10, 20, 30]])

        tlwh = tlbr2tlwh(tlbr)

        self.assertEqual(tlwh.shape, (1, 4))
        self.assertListEqual(tlwh.tolist(), [[10, 10, 10, 20]])

    def test_tlbr2ctwh(self):
        tlbr = torch.tensor([[10, 10, 20, 30]])

        ctwh = tlbr2ctwh(tlbr)

        self.assertEqual(ctwh.shape, (1, 4))
        self.assertListEqual(ctwh.tolist(), [[15, 20, 10, 20]])
