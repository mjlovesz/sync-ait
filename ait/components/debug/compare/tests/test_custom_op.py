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
import pytest
import shutil
import onnx

from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
from msquickcmp.npu.npu_dump_data import NpuDumpData
from msquickcmp.npu.npu_dump_data_bin2npy import data_convert
from msquickcmp.cmp_process import cmp_process
from msquickcmp.common import utils

@pytest.fixture(scope="module", autouse=True)
def args() -> None:
    if os.path.exists("./dump_data"):      
        shutil.rmtree("./dump_data")
    if os.path.exists("./model"):
        shutil.rmtree("./model")    
    if os.path.exists("./input"):
         shutil.rmtree("./input")

    cmp_args = CmpArgsAdapter(gold_model="./onnx/model.onnx",
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
    yield cmp_args

def test_init_onnx_dump_data(args):

    golden_dump = OnnxDumpData(args)
    golden_dump.generate_inputs_data()

    assert 'before_custom_op_model.onnx' in os.listdir('./model')
    assert 'after_custom_op_model.onnx' in os.listdir('./model')

    assert golden_dump.inputs_map['boxes_all'].shape == (1000,80,4)

def test_onnx_dump_data(args):

    golden_dump = OnnxDumpData(args)
    golden_dump.generate_inputs_data()

    # 2. generate npu dump data
    npu_dump = NpuDumpData(args, "./om/model.json")
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data(True)

    # 3. convert data from bin to npy if --convert is used
    npu_dump_path = data_convert(npu_dump_data_path, npu_net_output_data_path, args)
    print(npu_dump_path)

    # generate dump data by golden model
    golden_dump_data_path = golden_dump.generate_dump_data(npu_dump_path)
    assert len(os.listdir(golden_dump_data_path)) == 11

    golden_net_output_info = golden_dump.get_net_output_info()
    assert len(golden_net_output_info) == 3

def test_before_custom_op_dump_not_support(args):

    args.custom_op = ""
    golden_dump = OnnxDumpData(args)

    with pytest.raises(Exception) as error:
        golden_dump.generate_inputs_data()
