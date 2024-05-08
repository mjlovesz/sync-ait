import unittest

import torch

from components.tensor_view.ait_tensor_view.atb import read_atb_data, write_atb_data


class TestWriteReadAtbData(unittest.TestCase):
    def test_base_case(self):
        tensor = torch.rand(4, 4, 4, 4)

        write_atb_data(tensor, "./test.bin")
        actual_tensor = read_atb_data("./test.bin")
        assert torch.equal(tensor, actual_tensor)

    def test_with_dtype(self):
        tensor = torch.rand(4, 4, 4, 4, 4, dtype=torch.float16)

        write_atb_data(tensor, "./test-dtype.bin")
        actual_tensor = read_atb_data("./test-dtype.bin")
        assert torch.equal(tensor, actual_tensor)

    def test_with_dims(self):
        tensor = torch.rand(4, 4, 4, 4, 4, dtype=torch.float16)
        tensor.reshape(16, 16, 4, 1)

        write_atb_data(tensor, "./test-dims.bin")
        actual_tensor = read_atb_data("./test-dims.bin")
        assert torch.equal(tensor, actual_tensor)
