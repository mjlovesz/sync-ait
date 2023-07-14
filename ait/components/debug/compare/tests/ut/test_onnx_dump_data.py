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

import pytest
import torch
import numpy as np

from msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
from msquickcmp.common.utils import AccuracyCompareException


FAKE_ONNX_MODEL_PATH = "fake_msquickcmp_test_onnx_model.onnx"
OUT_PATH = FAKE_ONNX_MODEL_PATH.replace(".onnx", "")
INPUT_SHAPE = (1, 3, 32, 32)


class Args:
    def __init__(self, **kwargs):
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)


@pytest.fixture(scope="module", autouse=True)
def width_model():
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


@pytest.fixture(scope="function")
def fake_arguments():
    return Args(
        model_path=FAKE_ONNX_MODEL_PATH,
        out_path=OUT_PATH,
        input_path="",
        input_shape="",
        dym_shape_range="",
        custom_op="",
        onnx_fusion_switch=False,
        dump=True,
    )


def test_init_given_valid_when_any_then_pass(fake_arguments):
    aa = OnnxDumpData(fake_arguments)

    assert aa.origin_model is not None
    assert aa.origin_model is aa.model_with_inputs


def test_init_given_custom_op_when_valid_then_pass(fake_arguments):
    fake_arguments.custom_op = "/3/GlobalAveragePool"
    aa = OnnxDumpData(fake_arguments)

    assert aa.model_before_custom_op is not None
    assert aa.model_after_custom_op is not None
    assert aa.model_before_custom_op is aa.model_with_inputs


def test_init_given_model_path_when_not_exists_then_error(fake_arguments):
    fake_arguments.model_path = "not_exists_msquickcmp_test_onnx_model.onnx"
    with pytest.raises(AccuracyCompareException):
        OnnxDumpData(fake_arguments)


def test_init_given_model_path_when_not_onnx_then_error(fake_arguments):
    fake_arguments.model_path = fake_arguments.model_path.replace(".onnx", ".om")
    with pytest.raises(AccuracyCompareException):
        OnnxDumpData(fake_arguments)


def test_generate_inputs_data_given_random_when_valid_then_pass(fake_arguments):
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()

    assert aa.inputs_map is not None
    assert len(aa.inputs_map) == 1
    assert list(aa.inputs_map.values())[0].shape == INPUT_SHAPE


def test_generate_inputs_data_given_input_path_when_valid_then_pass(fake_arguments):
    input_path = os.path.join(OUT_PATH, "input_0.bin")
    input_data = np.random.uniform(size=INPUT_SHAPE).astype("float32")
    input_data.tofile(input_path)
    fake_arguments.input_path = input_path
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()

    assert aa.inputs_map is not None
    assert len(aa.inputs_map) == 1
    assert np.allclose(list(aa.inputs_map.values())[0], input_data, atol=1e-7)


def test_generate_inputs_data_given_input_shape_when_valid_then_pass(fake_arguments):
    fake_arguments.input_shape = "input.1:1,3,32,32"
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()

    assert aa.inputs_map is not None
    assert len(aa.inputs_map) == 1


def test_generate_inputs_data_given_input_path_when_not_equal_then_error(fake_arguments):
    input_path = os.path.join(OUT_PATH, "input_0.bin")
    fake_arguments.input_path = ",".join([input_path, input_path])
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_input_path_when_not_exists_then_error(fake_arguments):
    input_path = os.path.join(OUT_PATH, "not_exists_input_0.bin")
    fake_arguments.input_path = input_path
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_input_path_when_shape_not_match_then_error(fake_arguments):
    input_path = os.path.join(OUT_PATH, "input_0.bin")
    input_data = np.random.uniform(size=(1,)).astype("float32")
    input_data.tofile(input_path)
    fake_arguments.input_path = input_path
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_input_shape_when_shape_not_match_then_error(fake_arguments):
    fake_arguments.input_shape = "input.1:1,3,32"
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_input_shape_when_invalid_name_then_error(fake_arguments):
    fake_arguments.input_shape = "fake_input.1:1,3,32,32"
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data()


def test_generate_inputs_data_given_use_aipp_when_npu_dump_data_path_none_then_error(fake_arguments):
    aa = OnnxDumpData(fake_arguments)
    with pytest.raises(AccuracyCompareException):
        aa.generate_inputs_data(use_aipp=True)


def test_generate_dump_data_given_valid_when_any_then_pass(fake_arguments):
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()
    onnx_dump_data_dir = aa.generate_dump_data()

    assert onnx_dump_data_dir.endswith("onnx") or onnx_dump_data_dir.endswith("onnx/")
    assert os.path.exists(onnx_dump_data_dir)
    assert len(os.listdir(onnx_dump_data_dir)) > 0


def test_generate_dump_data_given_custom_op_when_valid_then_pass(fake_arguments):
    fake_arguments.custom_op = "/3/GlobalAveragePool"
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()

    fake_dump_data_path = "ReduceMeanD._3_GlobalAveragePool.time.output.0.npy"
    input_data = np.random.uniform(size=(1, 32, 1, 1)).astype("float32")
    np.save(os.path.join(OUT_PATH, fake_dump_data_path), input_data)
    om_parser = Args()
    om_parser.get_dynamic_scenario_info = lambda: (None, None)
    onnx_dump_data_dir = aa.generate_dump_data(npu_dump_path=OUT_PATH, om_parser=om_parser)


def test_generate_dump_data_given_custom_op_when_npu_dump_data_path_none_then_error(fake_arguments):
    fake_arguments.custom_op = "/3/GlobalAveragePool"
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()
    with pytest.raises(AccuracyCompareException):
        aa.generate_dump_data()


def test_generate_dump_data_given_custom_op_when_not_match_then_error(fake_arguments):
    fake_arguments.custom_op = "/4/Flatten"
    aa = OnnxDumpData(fake_arguments)
    aa.generate_inputs_data()
    om_parser = Args()
    om_parser.get_dynamic_scenario_info = lambda: (None, None)
    with pytest.raises(AccuracyCompareException):
        aa.generate_dump_data(npu_dump_path=OUT_PATH, om_parser=om_parser)
