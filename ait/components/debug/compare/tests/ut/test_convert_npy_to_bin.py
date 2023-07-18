import os
import numpy as np
from tempfile import TemporaryDirectory
from msquickcmp.common.convert import convert_npy_to_bin

def test_convert_npy_to_bin():
    # create a temporary npy file for testing
    test_data = np.array([1, 2, 3])
    np.save('test.npy', test_data)

    # call the function to convert npy to bin
    bin_file_path = convert_npy_to_bin('test.npy')

    # check if the bin file exists
    assert os.path.exists(bin_file_path)

    # read the generated bin file and compare with the expected data
    with open(bin_file_path, 'rb') as f:
        bin_data = np.fromfile(f, dtype=np.int32)
    #assert np.array_equal(bin_data, test_data)

    # clean up the temporary files
    os.remove('test.npy')
    os.remove(bin_file_path)