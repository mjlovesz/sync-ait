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
            --soc_version={}, --insert_op_conf={}'.format(fake_dym_shape_onnx_model, 
                                                          FAKE_DYM_SHAPE_OM_MODEL_PATH.replace(".om", ""),
                                                          acl.get_soc_name(),
                                                          "./test_resource/aipp.config")
        subprocess.run(cmd.split(), shell=False)

    yield FAKE_DYM_SHAPE_OM_MODEL_PATH
