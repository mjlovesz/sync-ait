# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import os
import numpy as np
from llm.common.tool import TensorBinFile, read_atb_data

# Mocked binary data for testing purposes
MOCKED_BINARY_DATA = b"$Version=1.0\n$Object.Count=1\n$Object.Length=4\nformat=2\ndtype=1\n" \
                      "dims=1,1,2\n$Object.data=0,4\n$End=1\n1212"
UNSUPPORT_DTYPE_BINARY_DATA = b"$Version=1.0\n$Object.Count=1\n$Object.Length=4\nformat=2\ndtype=999\n" \
                      "dims=1,1,2\n$Object.data=0,4\n$End=1\n"

@pytest.fixture
def create_mocked_bin_file(tmp_path):
    # Create a temporary bin file with mocked data for testing
    bin_file_path = tmp_path / "mocked_file.bin"
    with open(bin_file_path, "wb") as f:
        f.write(MOCKED_BINARY_DATA)
    return bin_file_path


@pytest.fixture
def create_error_bin_file(tmp_path):
    # Create a temporary bin file with mocked data for testing
    bin_file_path = tmp_path / "error_file.bin"
    with open(bin_file_path, "wb") as f:
        f.write(UNSUPPORT_DTYPE_BINARY_DATA)
    return bin_file_path


def test_tensor_bin_file_create_and_get_data(create_mocked_bin_file):
    bin_file = TensorBinFile(create_mocked_bin_file)
    data = bin_file.get_data()

    # Asserting the expected values based on the mocked data
    expected_shape = (1, 1, 2)
    expected_dtype = np.float16
    assert data.shape == expected_shape
    assert data.dtype == expected_dtype


def test_read_atb_data_valid_bin_file(create_mocked_bin_file):
    data = read_atb_data(create_mocked_bin_file)

    # Asserting the expected values based on the mocked data
    expected_shape = (1, 1, 2)
    expected_dtype = np.float16
    assert data.shape == expected_shape
    assert data.dtype == expected_dtype


def test_read_atb_data_invalid_file_extension(tmp_path):
    # Create a temporary file with an invalid extension for testing
    invalid_file_path = tmp_path / "invalid_file.txt"
    with open(invalid_file_path, "w") as f:
        f.write("Some random text")

    with pytest.raises(ValueError):
        read_atb_data(invalid_file_path)


def test_tensor_bin_file_unsupported_dtype():
    # Test scenario when an unsupported dtype is encountered
    invalid_data = b"$Version=1.0\n$Object.Count=1\n$Object.Length=4\nformat=2\ndtype=999\n" \
                      "dims=1,1,2\n$Object.data=0,4\n$End=1\n"
    with pytest.raises(ValueError):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(invalid_data)
            invalid_bin_file = f.name
            TensorBinFile(invalid_bin_file)


def test_read_atb_data_invalid_file_path():
    # Test scenario when an invalid file path is provided
    invalid_file_path = "/path/to/nonexistent_file.bin"
    with pytest.raises(ValueError):
        read_atb_data(invalid_file_path)


def test_read_atb_data_valid_non_bin_file(tmp_path):
    # Test scenario when a valid file with a non-bin extension is provided
    non_bin_file_path = tmp_path / "valid_file.txt"
    with open(non_bin_file_path, "w") as f:
        f.write("This is a valid text file.")

    with pytest.raises(ValueError):
        read_atb_data(non_bin_file_path)

