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
import os
import shutil
import json

import pytest
import numpy as np
import pandas as pd

from llm.compare import acc_cmp
from llm.compare import torchair_utils


FAKE_GOLDEN_DATA_PATH = "test_acc_cmp_fake_golden_data.npy"
FAKE_MY_DATA_PATH = "test_acc_cmp_fake_test_data.npy"


@pytest.fixture(scope='module', autouse=True)
def set_fake_parse_torchair_bin_dump_data():
    def fake_parse_torchair_bin_dump_data(my_path):
        return [np.ones([2, 3]).astype(np.float32)], [np.ones([2, 3]).astype(np.float32)]
    setattr(torchair_utils, "parse_torchair_bin_dump_data", fake_parse_torchair_bin_dump_data)


@pytest.fixture(scope='module')
def golden_data_file():
    golden_data = np.ones((2, 3)).astype(np.float32)
    np.save(FAKE_GOLDEN_DATA_PATH, golden_data)

    yield FAKE_GOLDEN_DATA_PATH

    if os.path.exists(FAKE_GOLDEN_DATA_PATH):
        os.remove(FAKE_GOLDEN_DATA_PATH)


@pytest.fixture(scope='module')
def test_data_file():
    test_data = np.ones((2, 3)).astype(np.float32)
    np.save(FAKE_MY_DATA_PATH, test_data)

    yield FAKE_MY_DATA_PATH

    if os.path.exists(FAKE_MY_DATA_PATH):
        os.remove(FAKE_MY_DATA_PATH)


@pytest.fixture(scope='module')
def test_dat_path():
    test_data_path = "test_acc_cmp_fake_test_data.dat"  # No need to create actual file

    yield test_data_path

    if os.path.exists(test_data_path):
        os.remove(test_data_path)


@pytest.fixture(scope='module')
def test_metadata_path():
    test_metadata_path = "test_acc_cmp_fake_metadata"
    if not os.path.exists(test_metadata_path):
        os.makedirs(test_metadata_path, mode=0o750)
    metadata = {0: {0: [FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH]}}

    metadata_path = os.path.join(test_metadata_path, "metadata.json")
    with open(metadata_path, "w") as ff:
        json.dump(metadata, ff)

    yield metadata_path

    if os.path.exists(test_metadata_path):
        shutil.rmtree(test_metadata_path)


def test_check_tensor_given_golden_data_when_nan_then_false():
    result, message = acc_cmp.check_tensor(np.zeros([2]).astype("float32") + np.nan, np.zeros([2]).astype("float32"))
    assert result is False and len(message) > 0 and "golden" in message.lower()


def test_fill_row_data_given_my_path_when_valid_then_pass(golden_data_file, test_data_file):
    row_data = acc_cmp.fill_row_data(0, 0, golden_data_file, test_data_file)
    assert isinstance(row_data, dict) and len(row_data) == 19
    assert row_data["cosine_similarity"] == '1.000000'
    assert len(row_data["cmp_fail_reason"]) == 0


def test_fill_row_data_given_loaded_my_data_when_valid_then_pass(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros_like(golden_data)
    row_data = acc_cmp.fill_row_data(0, 0, golden_data_file, my_path="test", loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 19
    assert row_data["cosine_similarity"] == 'NaN'
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_my_path_when_dir_then_error(golden_data_file):
    row_data = acc_cmp.fill_row_data(0, 0, golden_data_file, my_path="/")
    assert isinstance(row_data, dict) and len(row_data) == 5
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_golden_data_path_when_empty_then_error(test_data_file):
    row_data = acc_cmp.fill_row_data(0, 0, golden_data_path="", my_path=test_data_file)
    assert isinstance(row_data, dict) and len(row_data) == 5
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_my_path_when_nan_then_error(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros_like(golden_data) + np.nan
    row_data = acc_cmp.fill_row_data(0, 0, golden_data_file, my_path="test", loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 15
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_my_path_when_shape_not_match_then_error(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros([])
    row_data = acc_cmp.fill_row_data(0, 0, golden_data_file, my_path="test", loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 15
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_torchair_given_golden_data_path_when_valid_then_pass(golden_data_file):
    golden_data_path = {"inputs": [golden_data_file], "outputs": [golden_data_file]}
    result = acc_cmp.fill_row_data_torchair(0, 0, golden_data_path, my_path="test")
    assert len(result) == 2 and len(result[0]) == 19 and len(result[1]) == 19
    assert result[0]["cosine_similarity"] == "1.000000" and result[1]["cosine_similarity"] == "1.000000"


def test_save_compare_dataframe_to_csv_given_data_frame_when_valid_then_pass():
    dd = pd.DataFrame([{"aa": 11}, {"bb": 12}])
    csv_save_path = acc_cmp.save_compare_dataframe_to_csv(dd)
    assert os.path.exists(csv_save_path) and os.path.getsize(csv_save_path) > 0

    
def test_acc_compare_given_data_file_when_valid_then_pass(golden_data_file, test_data_file):
    acc_cmp.acc_compare(golden_data_file, test_data_file)


def test_read_data_given_data_file_when_valid_npy_then_pass(golden_data_file, test_data_file):
    data = acc_cmp.read_data(test_data_file)
    golden = np.load(golden_data_file)
    assert (data == golden).all()


def test_read_data_given_data_file_when_invalid_type_then_error(test_dat_path):
    with pytest.raises(TypeError):
        acc_cmp.read_data(test_dat_path)


def test_compare_data_given_data_file_when_valid_then_pass(golden_data_file, test_data_file):
    test_data = acc_cmp.read_data(test_data_file)
    golden_data = acc_cmp.read_data(golden_data_file)
    res = acc_cmp.compare_data(test_data, golden_data)
    assert res == {'cosine_similarity': '1.000000', 'max_relative_error': 0.0, 'mean_relative_error': 0.0,
                   'relative_euclidean_distance': 0.0, 'cmp_fail_reason': ''}


def test_compare_file_given_data_file_when_valid_then_pass(golden_data_file, test_data_file):
    res = acc_cmp.compare_file(golden_data_file, test_data_file)
    assert res == {'cosine_similarity': '1.000000', 'max_relative_error': 0.0, 'mean_relative_error': 0.0,
                   'relative_euclidean_distance': 0.0, 'cmp_fail_reason': ''}


def test_compare_metadata_given_golden_path_when_valid_then_pass(test_metadata_path):
    csv_save_path = acc_cmp.compare_metadata(test_metadata_path, output_path=".")
    assert os.path.exists(csv_save_path) and os.path.getsize(csv_save_path) > 0
