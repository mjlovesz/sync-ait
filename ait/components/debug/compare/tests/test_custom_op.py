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

import json
import onnx 
from google.protobuf.json_format import MessageToJson, Parse
import pytest
import acl

from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
from msquickcmp.npu.npu_dump_data import NpuDumpData
from msquickcmp.npu.npu_dump_data_bin2npy import data_convert
from msquickcmp.cmp_process import cmp_process
from msquickcmp.common import utils


@pytest.fixture(scope="session", autouse=True)
def fake_onnx_dir():
    os.makedirs("./onnx", exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def fake_onnx_model(fake_onnx_dir):
    with open("./test_resource/onnx/model_BatchMultiClassNMS.json", "r") as fi:
        onnx_json = json.loads(fi.read())
        onnx_str = json.dumps(onnx_json)
        convert_model = Parse(onnx_str, onnx.ModelProto())
        onnx.save(convert_model, "./onnx/model.onnx")


@pytest.fixture(scope="session", autouse=True)
def fake_om_model(fake_onnx_model):
    subprocess.run(('atc --model=./onnx/model.onnx --framework=5 \
                   --output=./om/model --soc_version=Ascend310' + acl.get_soc_name()).split(), shell=False)


@pytest.fixture(scope="session", autouse=True)
def cmp_args(fake_onnx_model, fake_om_model) -> None:
    if os.path.exists("./dump_data"):      
        shutil.rmtree("./dump_data")
    if os.path.exists("./model"):
        shutil.rmtree("./model")    
    if os.path.exists("./input"):
        shutil.rmtree("./input")

    args_adapter = CmpArgsAdapter(gold_model="./onnx/model.onnx",
                              om_model="./om/model.om",
                              input_data_path = "",
                              cann_path="/usr/local/Ascend/ascend-toolkit/latest/",
                              out_path="",
                              input_shape="boxes_all:1000,80,4;scores_all:1000,80",
                              device='0',
                              output_size="",
                              output_nodes="",
                              advisor=False,
                              dym_shape_range="",
                              dump=True,
                              bin2npy=True,
                              custom_op="BatchMultiClassNMS_1203")
    yield args_adapter


def test_init_onnx_dump_data(cmp_args):

    golden_dump = OnnxDumpData(cmp_args)
    golden_dump.generate_inputs_data("", False)

    assert 'before_custom_op_model.onnx' in os.listdir('./model')
    assert 'after_custom_op_model.onnx' in os.listdir('./model')
    assert golden_dump.inputs_map['boxes_all'].shape == (1000, 80, 4)


def test_onnx_dump_data(cmp_args):

    golden_dump = OnnxDumpData(cmp_args)
    golden_dump.generate_inputs_data("", False)

    # 2. generate npu dump data
    npu_dump = NpuDumpData(cmp_args, "./test_resource/om/model.json")
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data(True)

    # 3. convert data from bin to npy if --convert is used
    npu_dump_path = data_convert(npu_dump_data_path, npu_net_output_data_path, cmp_args)

    # generate dump data by golden model
    golden_dump_data_path = golden_dump.generate_dump_data(npu_dump_path)
    assert len(os.listdir(golden_dump_data_path)) == 14

    golden_net_output_info = golden_dump.get_net_output_info()
    assert len(golden_net_output_info) == 3


def test_before_custom_op_dump_not_support(cmp_args):

    cmp_args.custom_op = ""
    golden_dump = OnnxDumpData(cmp_args)

    with pytest.raises(Exception) as error:
        golden_dump.generate_inputs_data("", False)
