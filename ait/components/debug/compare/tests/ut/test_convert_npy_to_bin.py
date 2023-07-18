import unittest
import os
import numpy as np
import argparse
from msquickcmp.common.convert import convert_npy_to_bin

class TestConvertNpyToBin(unittest.TestCase):
    def setUp(self):
        self.npy_path = 'convert_test.npy'
        self.bin_path = 'convert_test.bin'
        self.args = argparse.Namespace(input_path=self.npy_path)

    def tearDown(self):
        if os.path.exists(self.npy_path):
            os.remove(self.npy_path)
        if os.path.exists(self.bin_path):
            os.remove(self.bin_path)

    def test_convert_npy_to_bin(self):
        # create a test npy file
        npy_data = np.array([1, 2, 3])
        np.save(self.npy_path, npy_data)

        # call the function to convert npy to bin
        convert_npy_to_bin(self.args.input_path)

        # check if the bin file is generated
        assert os.path.exists(self.bin_path)

    def test_convert_npy_to_bin_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            convert_npy_to_bin('nonexistent_file.npy')

    def test_convert_npy_to_bin_non_npy_file(self):
        input_path = 'test.txt'
        outputs = convert_npy_to_bin(input_path)
        self.assertEqual(outputs, input_path)

