# Copyright 2022 Huawei Technologies Co., Ltd
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

import pathlib

import click


opt_gold_model = click.option(
    '-gm',
    '--gold-model',
    'gold_model',
    required=True,
    help="The original model (.onnx or .pb) file path",
)

opt_om_model = click.option(
    "-om",
    "--om-model",
    "om_model",
    required=True,
    help="The offline model (.om) file path",
)

opt_input = click.option(
    "-i",
    "--input",
    "input",
    default="",
    help="The input data path of the model. Separate multiple inputs with commas(,)."
    " E.g: input_0.bin,input_1.bin"
)

opt_cann_path = click.option(
    "-c",
    "--cann-path",
    "cann_path",
    default="/usr/local/Ascend/ascend-toolkit/latest/",
    help="The CANN installation path"
)

opt_out_path = click.option(
    "-o",
    "--out-path",
    "out_path",
    default="",
    help="The output path"
)

opt_input_shape = click.option(
    "-s",
    "--input-shape",
    "input_shape",
    default="",
    help="Shape of input shape. Separate multiple nodes with semicolons(;)."
         "E.g: input_name1:1,224,224,3;input_name2:3,300"
)

opt_device = click.option(
    "-d",
    "--device",
    "device",
    default="0",
    help="Input device ID [0, 255], default is 0."
)

opt_output_size = click.option(
    "--output-size",
    "output_size",
    default="",
    help="The size of output. Separate multiple sizes with commas(,). E.g: 10200,34000"
)

opt_output_nodes = click.option(
    "--output-nodes",
    "output_nodes",
    default="",
    help="Output nodes designated by user. Separate multiple nodes with semicolons(;)."
         "E.g: node_name1:0;node_name2:1;node_name3:0"
)

opt_advisor = click.option(
    "--advisor",
    "advisor",
    is_flag=True,
    help="Enable advisor after compare."
)
