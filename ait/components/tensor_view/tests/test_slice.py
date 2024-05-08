import unittest

import torch

from ait.components.tensor_view.ait_tensor_view.operation import SliceOperation


class TestSliceOperation(unittest.TestCase):
    def test_single_dimension(self):
        tensor = torch.rand(4, 4, 4)
        op1 = SliceOperation("[3]")
        op2 = SliceOperation("[:3]")
        op3 = SliceOperation("[1:2:]")
        op4 = SliceOperation("[::2]")
        assert torch.equal(op1.process(tensor), tensor[3])
        assert torch.equal(op2.process(tensor), tensor[:3])
        assert torch.equal(op3.process(tensor), tensor[1:2:])
        assert torch.equal(op4.process(tensor), tensor[::2])

    def test_ellipsis(self):
        tensor = torch.rand(3, 4, 5)
        op = SliceOperation("[..., 1]")
        assert torch.equal(op.process(tensor), tensor[..., 1])

    def test_multi_dimension(self):
        tensor = torch.rand(3, 4, 5)
        assert torch.equal(SliceOperation("[1, 2:4, 0]").process(tensor), tensor[1, 2:4, 0])

    def test_invalid_dimension(self):
        tensor = torch.rand(3, 4)
        with self.assertRaises(ValueError):
            op = SliceOperation("[1, 2, 3]")
            op.process(tensor)

    def test_index_out_of_range(self):
        tensor = torch.rand(3, 4)
        with self.assertRaises(IndexError):
            op = SliceOperation("[3, 0]")
            op.process(tensor)
