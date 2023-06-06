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
from msquickcmp.npu.om_parser import OmParser


def test_dynamic_scenario():
    NpuDumpData


def test_get_shape_size():
    om_parser = OmParser("./test_resource/om/model.json")
    shape_size_array = om_parser.get_shape_size()
    assert shape_size_array == [1280000, 320000]


def test_net_output_count():
    om_parser = OmParser("./test_resource/om/model.json")

    count = om_parser.get_net_output_count()
    assert count == 3

    atc_cmd = om_parser.get_atc_cmdline()
    assert "model" in atc_cmd

    net_output = om_parser.get_expect_net_output_name()
    assert len(net_output) == 3
    assert (net_output[0]) == "Cast_1219:0:output0"
    assert (net_output[1]) == 'PartitionedCall_Gather_1221_gatherv2_3:0:output2'
    assert (net_output[2]) == 'Reshape_1213:0:output1'


def test_get_atc_cmdline():
    om_parser = OmParser("./test_resource/om/model.json")
    atc_cmd = om_parser.get_atc_cmdline()
    assert "model" in atc_cmd

def test_get_expect_net_output_name():
    om_parser = OmParser("./test_resource/om/model.json")
    net_output = om_parser.get_expect_net_output_name()
    assert len(net_output) == 3
    assert (net_output[0]) == "Cast_1219:0:output0"
    assert (net_output[1]) == 'PartitionedCall_Gather_1221_gatherv2_3:0:output2'
    assert (net_output[2]) == 'Reshape_1213:0:output1'


