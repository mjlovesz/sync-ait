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

import subprocess
import os
import stat

import pytest
import torch
import acl

from msquickcmp.common import utils
from msquickcmp.common.utils import parse_input_shape_to_list
from msquickcmp.npu.om_parser import OmParser
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.atc.atc_utils import AtcUtils
from msquickcmp.cmp_process import cmp_process

WRITE_FLAGS = os.O_WRONLY | os.O_CREAT  # 注意根据具体业务的需要设置文件读写方式
WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR  # 注意根据具体业务的需要设置文件权限


class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.linear1 = torch.nn.Linear(16*7*7, 10)
    
    def forward(self, x):
        y_pred = self.linear1(self.conv1(x).reshape(1, -1))
        return y_pred


@pytest.fixture(scope="session", autouse=True)
def fake_tmp_dir():
    os.makedirs("./tmp", exist_ok=True)


@pytest.fixture(scope="module", autouse=True)
def fake_onnx_model(fake_tmp_dir):
    model_path = "./tmp/fake.onnx"
    model = TwoLayerNet()
    dummy_input = torch.rand(1, 3, 7, 7, dtype=torch.float)
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["image"]
    )


@pytest.fixture(scope="module", autouse=True)
def fake_aipp_config(fake_tmp_dir):
    fake_aipp_config_path = "./tmp/aipp.config"
    data = """aipp_op{
aipp_mode:static
input_format : RGB888_U8

src_image_size_w : 10
src_image_size_h : 10

crop: true
load_start_pos_h : 1
load_start_pos_w : 1
crop_size_w : 7
crop_size_h: 7

min_chn_0 : 123.675
min_chn_1 : 116.28
min_chn_2 : 103.53
var_reci_chn_0: 0.0171247538316637
var_reci_chn_1: 0.0175070028011204
var_reci_chn_2: 0.0174291938997821
}
"""

    with os.fdopen(os.open(fake_aipp_config_path, WRITE_FLAGS, WRITE_MODES), 'w') as fout:
        fout.write(data)
    return fake_aipp_config_path


@pytest.fixture(scope="module", autouse=True)
def fake_switch_config(fake_tmp_dir):
    fake_switch_config_path = "./tmp/fusionswitch.cfg"
    data = """{
    "Switch":{
        "GraphFusion":{
            "ALL":"off"
        },
        "UBFusion":{
            "ALL":"off"
        }
    }
}
    """
    
    with os.fdopen(os.open(fake_switch_config_path, WRITE_FLAGS, WRITE_MODES), 'w') as fout:
        fout.write(data)


@pytest.fixture(scope="module", autouse=True)
def fake_om_model(fake_onnx_model, fake_aipp_config, fake_switch_config):
    cmd = 'atc --model ./tmp/fake.onnx --soc_version '  + acl.get_soc_name() + \
        '--framework 5 --input_format NCHW --input_shape image:1,3,7,7 --output ./tmp/fake --insert_op_conf' + \
        './tmp/aipp.config --fusion_switch_file ./tmp/fusionswitch.cfg'

    subprocess.run(cmd.split(), shell=False)
    return "./tmp/fake.om"


def test_aipp_function_st_pass(fake_om_model):
    args = CmpArgsAdapter(gold_model="tmp/fake.onnx",
                          om_model="tmp/fake.om",
                          input_data_path = "",
                          cann_path="/usr/local/Ascend/ascend-toolkit/latest/",
                          out_path="./tmp/",
                          input_shape="",
                          device="0",
                          output_size="",
                          output_nodes="",
                          advisor=False,
                          dym_shape_range="",
                          dump=True,
                          bin2npy=False)
    cmp_process(args, use_cli=True)


def test_aipp_function_st_no_dump_error(fake_om_model):
    args = CmpArgsAdapter(gold_model="tmp/fake.onnx",
                          om_model="tmp/fake.om",
                          input_data_path = "",
                          cann_path="/usr/local/Ascend/ascend-toolkit/latest/",
                          out_path="./tmp/",
                          input_shape="image:1,3,7,7",
                          device="0",
                          output_size="",
                          output_nodes="",
                          advisor=False,
                          dym_shape_range="",
                          dump=False,
                          bin2npy=False)
    with pytest.raises(utils.AccuracyCompareException) as error:
        cmp_process(args, use_cli=True)
    assert error.value.error_info == utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR


def test_parse_input_shape_to_list_when_wrong_format_error():
    with pytest.raises(utils.AccuracyCompareException) as error:
        parse_input_shape_to_list("image1,3,224,224")
    assert error.value.error_info == utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR


def test_get_aipp_config_content_pass(fake_om_model):
    args = CmpArgsAdapter(gold_model="tmp/fake.onnx",
                          om_model="tmp/fake.om",
                          input_data_path = "",
                          cann_path="/usr/local/Ascend/ascend-toolkit/latest/",
                          out_path="./tmp/",
                          input_shape="image:1,3,7,7",
                          device="0",
                          output_size="",
                          output_nodes="",
                          advisor=False,
                          dym_shape_range="",
                          dump=True,
                          bin2npy=False)
    output_json_path = AtcUtils(args).convert_model_to_json()
    om_parser = OmParser(output_json_path)
    assert om_parser.get_aipp_config_content()