import unittest

import torch

from ait.components.tensor_view.ait_tensor_view.operation import PermuteOperation


class TestPermuteOperation(unittest.TestCase):
    tensor = torch.rand(1, 2, 3, 4)

    def test_unequal_dimension(self):
        op = PermuteOperation("(3,4,2)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_duplicate(self):
        op = PermuteOperation("(2,3,2,4,0)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_not_n1(self):
        op = PermuteOperation("(0,1,2,3,5)")
        with self.assertRaises(ValueError):
            op.process(self.tensor)

    def test_valid(self):
        op = PermuteOperation("(3,1,2,0)")
        assert torch.equal(op.process(self.tensor), self.tensor.permute(3, 1, 2, 0))
