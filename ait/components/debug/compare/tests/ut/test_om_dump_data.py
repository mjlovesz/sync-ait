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
import subprocess

import pytest
import torch
import acl
import numpy as np

from msquickcmp.npu.npu_dump_data import NpuDumpData
from msquickcmp.common.utils import AccuracyCompareException, parse_input_shape_to_list

FAKE_DYM_SHAPE_ONNX_MODEL_PATH = "fake_dym_shape_test_onnx_model.onnx"
FAKE_DYM_SHAPE_OM_MODEL_PATH = "fake_dym_shape_test_onnx_model.om"
FAKE_DYM_SHAPE_OM_MODEL_JSON_PATH = "fake_dym_shape_test_onnx_model.json"

FAKE_OM_MODEL_WITH_AIPP_PATH = "fake_with_aipp_test_onnx_model.om"
FAKE_OM_MODEL_WITH_AIPP_JSON_PATH = "fake_with_aipp_test_onnx_model.json"

FAKE_OM_MODEL_PATH = "fake_test_onnx_model.om"
FAKE_OM_MODEL_JSON_PATH = "fake_test_onnx_model.json"
OM_OUT_PATH = FAKE_OM_MODEL_PATH.replace(".om", "")


FAKE_ONNX_MODEL_PATH = "fake_msquickcmp_test_onnx_model.onnx"
OUT_PATH = FAKE_ONNX_MODEL_PATH.replace(".onnx", "")
INPUT_SHAPE = (1, 3, 32, 32)


class Args:
    def __init__(self, **kwargs):
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)


@pytest.fixture(scope="function")
def fake_arguments():
    return Args(
        model_path=FAKE_OM_MODEL_PATH,
        out_path=OM_OUT_PATH,
        input_shape="",
        input_path="",
        dump=True,
        output_size="",
        device="0",
    )


@pytest.fixture(scope="module", autouse=True)
def width_onnx_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 1, 1),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 32, 3, 2),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 32),
        torch.nn.Linear(32, 10),
    )
    torch.onnx.export(model, torch.ones(INPUT_SHAPE), FAKE_ONNX_MODEL_PATH)
    yield FAKE_ONNX_MODEL_PATH

    if os.path.exists(FAKE_ONNX_MODEL_PATH):
        os.remove(FAKE_ONNX_MODEL_PATH)
    if os.path.exists(OUT_PATH):
        shutil.rmtree(OUT_PATH)

@pytest.fixture(scope="module", autouse=True)
def fake_dym_shape_onnx_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 1, 1),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 32, 3, 2),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 32),
        torch.nn.Linear(32, 10),
    )
    input_name = 'input0'
    input_data = torch.ones(INPUT_SHAPE)

    torch.onnx.export(model, 
                      input_data, 
                      FAKE_DYM_SHAPE_ONNX_MODEL_PATH,
                      input_names=[input_name],
                      dynamic_axes={input_name:{0:'bs'}})
    
    yield FAKE_DYM_SHAPE_ONNX_MODEL_PATH

    if os.path.exists(FAKE_DYM_SHAPE_ONNX_MODEL_PATH):
        os.remove(FAKE_DYM_SHAPE_ONNX_MODEL_PATH)


@pytest.fixture(scope="module", autouse=True)
def fake_om_model(width_onnx_model):
    if not os.path.exists(FAKE_OM_MODEL_PATH):
        cmd = 'atc --model={}, --framework=5 --output={}, \
            --soc_version={}'.format(width_onnx_model, 
                                     OM_OUT_PATH,
                                     acl.get_soc_name())
        subprocess.run(cmd.split(), shell=False)

    yield FAKE_OM_MODEL_PATH


@pytest.fixture(scope="module", autouse=True)
def fake_om_model_with_aipp(width_onnx_model):
    if not os.path.exists(FAKE_OM_MODEL_WITH_AIPP_PATH):
        cmd = 'atc --model={}, --framework=5 --output={} \
            --soc_version={}, --insert_op_conf={}'.format(width_onnx_model, 
                                                          FAKE_OM_MODEL_WITH_AIPP_PATH.replace(".om", ""),
                                                          acl.get_soc_name(),
                                                          "./test_resource/aipp.config")
        subprocess.run(cmd.split(), shell=False)

    yield FAKE_OM_MODEL_WITH_AIPP_PATH

@pytest.fixture(scope="module", autouse=True)
def fake_om_model_dym_shape(fake_dym_shape_onnx_model):
    if not os.path.exists(FAKE_DYM_SHAPE_OM_MODEL_PATH):
        cmd = 'atc --model={}, --framework=5 --output={} \
            --soc_version={}, --input_shape_range={}'.format(fake_dym_shape_onnx_model, 
                                                          FAKE_DYM_SHAPE_OM_MODEL_PATH.replace(".om", ""),
                                                          acl.get_soc_name(),
                                                          "./test_resource/aipp.config",
                                                          "input0:[1~2],3,32,32")
        subprocess.run(cmd.split(), shell=False)

    yield FAKE_DYM_SHAPE_OM_MODEL_PATH

def test_init_given_valid_when_any_then_pass(fake_arguments):
    aa = NpuDumpData(fake_arguments, False)

    except_net_output_node = aa.get_expect_output_name()

    assert len(except_net_output_node) == 1
    
    assert aa.om_parser is not None
    assert aa.dynamic_input is not None
    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)


def test_init_given_invalid_when_any_then_pass(fake_arguments):
    fake_arguments.model_path = ""

    with pytest.raises(AccuracyCompareException):
        aa = NpuDumpData(fake_arguments, False)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)
        
def test_generate_inputs_data_given_random_when_valid_then_pass(fake_arguments):

    npu_dump = NpuDumpData(fake_arguments, False)

    assert npu_dump.om_parser is not None
    assert npu_dump.dynamic_input is not None

    npu_dump.generate_inputs_data()

    assert os.path.exists(os.path.join(fake_arguments.out_path, "input"))

    inputs_list = parse_input_shape_to_list(fake_arguments.input_shape)
    input_bin_files = os.listdir(os.path.join(fake_arguments.out_path, "input"))

    for input_file, input_shape in zip(input_bin_files, inputs_list):
        input_data =  np.fromfile(input_file)
        assert np.prod(input_data.shape) == np.prod(input_shape)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)

def test_generate_inputs_data_given_input_path_when_valid_then_pass(fake_arguments):

    tmp_input_data = "tmp_input_data"
    if not os.path.exists(tmp_input_data):
        os.makedirs(tmp_input_data, mode=0o700)

    input_path = os.path.join(tmp_input_data, "input_0.bin")
    input_data = np.random.uniform(size=INPUT_SHAPE).astype("float32")
    input_data.tofile(input_path)
    fake_arguments.input_path = input_path

    npu_dump = NpuDumpData(fake_arguments, False)

    assert npu_dump.om_parser is not None
    assert npu_dump.dynamic_input is not None

    npu_dump.generate_inputs_data()

    assert os.path.exists(os.path.join(fake_arguments.out_path, "input"))

    inputs_list = parse_input_shape_to_list(fake_arguments.input_shape)
    input_bin_files = os.listdir(os.path.join(fake_arguments.out_path, "input"))

    for input_file, input_shape in zip(input_bin_files, inputs_list):
        input_data =  np.fromfile(input_file)
        assert np.prod(input_data.shape) == np.prod(input_shape)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)
    if os.path.exists(tmp_input_data):
        shutil.rmtree(tmp_input_data)

def test_generate_inputs_data_given_input_path_when_golden_then_pass(fake_arguments):

    tmp_input_data = "tmp_input_data"
    if not os.path.exists(tmp_input_data):
        os.makedirs(tmp_input_data, mode=0o700)

    input_path = os.path.join(tmp_input_data, "input_0.bin")
    input_data = np.random.uniform(size=INPUT_SHAPE).astype("float32")
    input_data.tofile(input_path)
    fake_arguments.input_path = input_path

    npu_dump = NpuDumpData(fake_arguments, True)

    assert npu_dump.om_parser is not None
    assert npu_dump.dynamic_input is not None

    npu_dump.generate_inputs_data()

    assert os.path.exists(os.path.join(fake_arguments.out_path, "input"))

    inputs_list = parse_input_shape_to_list(fake_arguments.input_shape)
    input_bin_files = os.listdir(os.path.join(fake_arguments.out_path, "input"))

    for input_file, input_shape in zip(input_bin_files, inputs_list):
        input_data =  np.fromfile(input_file)
        assert np.prod(input_data.shape) == np.prod(input_shape)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)
    if os.path.exists(tmp_input_data):
        shutil.rmtree(tmp_input_data)

def test_generate_inputs_data_given_random_data_when_aipp_then_pass(fake_arguments, fake_om_model_with_aipp):
    
    fake_arguments.model_path = fake_om_model_with_aipp
    fake_arguments.out_path = fake_om_model_with_aipp.replace(".om", "")

    npu_dump = NpuDumpData(fake_arguments, False)

    assert npu_dump.om_parser is not None
    assert npu_dump.dynamic_input is not None

    npu_dump.generate_inputs_data()

    assert os.path.exists(os.path.join(fake_arguments.out_path, "input"))

    inputs_list = parse_input_shape_to_list(fake_arguments.input_shape)
    input_bin_files = os.listdir(os.path.join(fake_arguments.out_path, "input"))

    for input_file, input_shape in zip(input_bin_files, inputs_list):
        input_data =  np.fromfile(input_file)
        assert np.prod(input_data.shape) == np.prod(input_shape)

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)


def test_generate_dump_data_given_random_data_when_valid_then_pass(fake_arguments):

    npu_dump = NpuDumpData(fake_arguments, False)
    npu_dump.generate_inputs_data()

    om_dump_data_dir = npu_dump.generate_dump_data()
    assert os.path.exists(om_dump_data_dir)

    assert len(os.listdir(om_dump_data_dir)) > 0

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)


def test_generate_dump_data_given_random_data_when_dump_false_then_pass(fake_arguments):

    fake_arguments.dump = False
    npu_dump = NpuDumpData(fake_arguments, False)
    npu_dump.generate_inputs_data()

    om_dump_data_dir = npu_dump.generate_dump_data()
    assert os.path.exists(om_dump_data_dir)

    assert len(os.listdir(om_dump_data_dir)) > 0

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)


def test_generate_dump_data_given_random_data_when_dym_shape_then_pass(fake_arguments,
                                                                       fake_om_model_dym_shape):

    fake_arguments.model_path = fake_om_model_dym_shape
    fake_arguments.out_path = fake_om_model_dym_shape.replace(".om", "")

    npu_dump = NpuDumpData(fake_arguments, False)
    npu_dump.generate_inputs_data()

    om_dump_data_dir = npu_dump.generate_dump_data()
    assert os.path.exists(om_dump_data_dir)

    assert len(os.listdir(om_dump_data_dir)) > 0

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)

def test_generate_dump_data_given_any_when_dym_shape_and_golden_then_pass(fake_arguments,
                                                                          fake_om_model_dym_shape):

    fake_arguments.model_path = fake_om_model_dym_shape
    fake_arguments.out_path = fake_om_model_dym_shape.replace(".om", "")

    fake_arguments.input_shape = "input0:2,3,32,32"
    npu_dump = NpuDumpData(fake_arguments, True)
    npu_dump.generate_inputs_data()

    om_dump_data_dir = npu_dump.generate_dump_data()
    assert os.path.exists(om_dump_data_dir)

    assert len(os.listdir(om_dump_data_dir)) > 0

    if os.path.exists(fake_arguments.out_path):
        shutil.rmtree(fake_arguments.out_path)