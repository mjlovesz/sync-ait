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
MOCKED_BINARY_DATA = (
    b"$Version=1.0\n"
    b"$Object.Count=1\n"
    b"$Object.Length=4\n"
    b"format=2\n"
    b"dtype=1\n"
    b"dims=1,1,2\n"
    b"$Object.data=0,4\n"
    b"$End=1\n"
    b"1212"
)


UNSUPPORT_DTYPE_BINARY_DATA = (
    b"$Version=1.0\n"
    b"$Object.Count=1\n"
    b"$Object.Length=4\n"
    b"format=2\n"
    b"dtype=999\n"
    b"dims=1,1,2\n"
    b"$Object.data=0,4\n"
    b"$End=1\n"
)


@pytest.fixture
def create_mocked_bin_file(tmp_path):
    # Create a temporary bin file with mocked data for testing
    bin_file_path = tmp_path / "mocked_file.bin"
    bin_file_path = str(bin_file_path)
    with open(bin_file_path, "wb") as f:
        f.write(MOCKED_BINARY_DATA)
    yield bin_file_path
    if os.path.exists(bin_file_path):
        os.remove(bin_file_path)


@pytest.fixture
def create_unsupport_dtype_bin_file(tmp_path):
    # Create a temporary bin file with mocked data for testing
    unsupport_dtype_file_path = tmp_path / "unsupport_dtype_file.bin"
    unsupport_dtype_file_path = str(unsupport_dtype_file_path)
    with open(unsupport_dtype_file_path, "wb") as f:
        f.write(UNSUPPORT_DTYPE_BINARY_DATA)
    yield unsupport_dtype_file_path
    if os.path.exists(unsupport_dtype_file_path):
        os.remove(unsupport_dtype_file_path)


@pytest.fixture
def create_invalid_format_file(tmp_path):
    invalid_file_path = tmp_path / "invalid_file.txt"
    invalid_file_path = str(invalid_file_path)
    with open(invalid_file_path, "w") as f:
        f.write("Some random text")
    yield invalid_file_path
    if os.path.exists(invalid_file_path):
        os.remove(invalid_file_path)


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


def test_read_atb_data_invalid_file_extension(create_invalid_format_file):
    # Create a temporary file with an invalid extension for testing
    with pytest.raises(ValueError):
        read_atb_data(create_invalid_format_file)


def test_tensor_bin_file_unsupported_dtype(create_unsupport_dtype_bin_file):
    # Test scenario when an unsupported dtype is encountered
    bin_file = TensorBinFile(create_unsupport_dtype_bin_file)
    with pytest.raises(ValueError):
        bin_file.get_data()

