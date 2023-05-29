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
import pytest
import click
import shutil

from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
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
                              input_shape="input0:1,3,1344,1344",
                              device=0,
                              output_size="",
                              output_nodes="",
                              advisor=False,
                              dym_shape_range="",
                              dump=True,
                              bin2npy=False,
                              custom_op="BatchMultiClassNMS_1203")
    yield cmp_args

def test_before_custom_op_dump(args):

    golden_dump = OnnxDumpData(args)
    golden_dump.generate_inputs_data()

    assert golden_dump.inputs_map['input0'].shape == (1, 3, 1344, 1344)

    # 4. generate dump data by golden model
    golden_dump_data_path = golden_dump.generate_dump_data()
    assert len(os.listdir(golden_dump_data_path)) == 904

    golden_net_output_info = golden_dump.get_net_output_info()
    assert len(golden_net_output_info) == 2



