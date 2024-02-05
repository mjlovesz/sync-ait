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
import stat
import shutil

import pytest
import numpy as np

from llm.compare import torchair_utils


FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_PBTXT_FILE_NAME = "test_torchair_utils_fake_pbtxt_file.txt"
FAKE_GE_DUMP_DATA_NAME = "test_torchair_utils_fake_ge_dump_data"
FAKE_FX_DUMP_DATA_NAME = "test_torchair_utils_fake_fx_dump_data"


@pytest.fixture(scope='module', autouse=True)
def fake_pbtxt_file():
    contents = """op {
      name: "Add_2"
      output_desc {
        name: "test"
        attr {
          key: "_fx_tensor_name"
          value {
            s: "mm-aten.mm.default.OUTPUT.0"
          }
        }
        attr {
          name: "tt2"
        }
      }
    }"""

    with os.fdopen(os.open(FAKE_PBTXT_FILE_NAME, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), 'w') as ff:
        ff.write(contents)

    yield

    if os.path.exists(FAKE_PBTXT_FILE_NAME):
        os.remove(FAKE_PBTXT_FILE_NAME)


@pytest.fixture(scope='module', autouse=True)
def fake_ge_dump_data():
    base_path = os.path.join(FAKE_GE_DUMP_DATA_NAME, "1")
    os.makedirs(base_path, mode=0o750, exist_ok=True)

    file_names = [
        "Add.Add_2.44.6.17065969121619", "Cast.Cast_9.19.6.17065969118878", "ConcatV2D.ConcatV2.42.6.17065969121611"
    ]
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        with os.fdopen(os.open(file_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), 'wb') as ff:
            pass

    ge_graph_path = os.path.join(FAKE_GE_DUMP_DATA_NAME, torchair_utils.GE_GRAPH_FILE_PREFIX + "_test.txt")
    with os.fdopen(os.open(ge_graph_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), 'wb') as ff:
        pass
    
    yield

    if os.path.exists(FAKE_GE_DUMP_DATA_NAME):
        shutil.rmtree(FAKE_GE_DUMP_DATA_NAME)


@pytest.fixture(scope='module', autouse=True)
def fake_fx_dump_data():
    base_path = os.path.join(FAKE_FX_DUMP_DATA_NAME, "1")
    os.makedirs(base_path, mode=0o750, exist_ok=True)

    file_names = [
        "mm-aten.mm.default.INPUT.0.20240125031118787351.npy",
        "mm-aten.mm.default.INPUT.1.20240125031118787351.npy",
        "mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy",
    ]
    for file_name in file_names:
        np.save(os.path.join(base_path, file_name), np.zeros([]))
    
    yield

    if os.path.exists(FAKE_FX_DUMP_DATA_NAME):
        shutil.rmtree(FAKE_FX_DUMP_DATA_NAME)


def test_get_torchair_ge_graph_path_given_path_when_valid_then_pass():
    ge_graph_path = torchair_utils.get_torchair_ge_graph_path(FAKE_GE_DUMP_DATA_NAME)
    assert ge_graph_path is not None
    assert os.path.basename(ge_graph_path).startswith(torchair_utils.GE_GRAPH_FILE_PREFIX)


def test_get_torchair_ge_graph_path_given_path_when_invalid_then_none():
    ge_graph_path = torchair_utils.get_torchair_ge_graph_path(FAKE_FX_DUMP_DATA_NAME)
    assert ge_graph_path is None


def test_parse_pbtxt_to_dict_given_path_when_valid_then_pass():
    result = torchair_utils.parse_pbtxt_to_dict(FAKE_PBTXT_FILE_NAME)
    assert isinstance(result, list) and isinstance(result[0], dict)
    expected_result = [{'op': {
        'name': 'Add_2',
        'output_desc': {
            'name': 'test',
            'attr': {'key': '_fx_tensor_name', 'value': {'s': 'mm-aten.mm.default.OUTPUT.0'}},
            'attr#1': {'name': 'tt2'}
        }
    }}]
    assert result == expected_result


def test_init_ge_dump_data_from_bin_path_given_path_when_valid_then_pass():
    result = torchair_utils.init_ge_dump_data_from_bin_path(FAKE_GE_DUMP_DATA_NAME)
    expected_result = {0: {}, 1: {
        'Add_2': os.path.join(FAKE_GE_DUMP_DATA_NAME, '1', 'Add.Add_2.44.6.17065969121619'),
        'Cast_9': os.path.join(FAKE_GE_DUMP_DATA_NAME, '1', 'Cast.Cast_9.19.6.17065969118878'),
        'ConcatV2': os.path.join(FAKE_GE_DUMP_DATA_NAME, '1', 'ConcatV2D.ConcatV2.42.6.17065969121611')
    }}
    assert result == expected_result


def test_init_fx_dump_data_from_path_given_path_when_valid_then_pass():
    result = torchair_utils.init_fx_dump_data_from_path(FAKE_FX_DUMP_DATA_NAME)
    expected_result = {1: {
        'mm-aten.mm.default': {
            'input': [
                os.path.join(FAKE_FX_DUMP_DATA_NAME, '1', 'mm-aten.mm.default.INPUT.0.20240125031118787351.npy'),
                os.path.join(FAKE_FX_DUMP_DATA_NAME, '1', 'mm-aten.mm.default.INPUT.1.20240125031118787351.npy')],
            'output': [
                os.path.join(FAKE_FX_DUMP_DATA_NAME, '1', 'mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy')
            ]
        }
    }}
    assert result == expected_result


def test_build_metadata_given_path_when_valid_then_pass():
    result = torchair_utils.build_metadata(FAKE_FX_DUMP_DATA_NAME, FAKE_GE_DUMP_DATA_NAME, FAKE_PBTXT_FILE_NAME)
    expected_result = {1: {1: [
        {
            'inputs': [
                'test_torchair_utils_fake_fx_dump_data/1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy',
                'test_torchair_utils_fake_fx_dump_data/1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy'],
            'outputs': ['test_torchair_utils_fake_fx_dump_data/1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy']
        },
        'test_torchair_utils_fake_ge_dump_data/1/Add.Add_2.44.6.17065969121619']
    }}
    assert result == expected_result
