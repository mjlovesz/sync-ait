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

import pathlib

import click

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.common.click_utils import default_off_knowledges, \
    validate_opt_converter, check_args, check_node_name

opt_knowledges = click.option(
    '-know',
    '--knowledges',
    'knowledges',
    default=','.join(
        knowledge
        for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
        if knowledge not in default_off_knowledges
    ),
    type=str,
    callback=check_args,
    help='Knowledges(index/name) you want to apply. Seperate by comma(,), Default to all except fix knowledges.'
)


opt_processes = click.option(
    '-p',
    '--processes',
    'processes',
    default=1,
    type=click.IntRange(1, 64),
    help='Use multiprocessing in evaluate mode, determine how many processes should be spawned. Default to 1'
)


opt_verbose = click.option(
    '-v',
    '--verbose',
    'verbose',
    is_flag=True,
    default=False,
    help='Show progress in evaluate mode. Default to false.'
)


opt_recursive = click.option(
    '-r',
    '--recursive',
    'recursive',
    is_flag=True,
    default=False,
    help='Process onnx in a folder recursively if any folder provided as PATH. Default to false.'
)


opt_output = click.option(
    '-of',
    '--output-file',
    'output_model',
    required=True,
    type=str,
    callback=check_args,
    help='Output onnx model name'
)


opt_input = click.option(
    '-in',
    '--input',
    'input_model',
    required=True,
    type=str,
    callback=check_args,
    help='Input onnx model to be optimized'
)


opt_graph1 = click.option(
    '-g1',
    '--graph1',
    'graph1',
    required=True,
    type=str,
    callback=check_args,
    help='First onnx model to be consolidated'
)


opt_graph2 = click.option(
    '-g2',
    '--graph2',
    'graph2',
    required=True,
    type=str,
    callback=check_args,
    help='Second onnx model to be consolidated'
)


opt_prefix = click.option(
    '-pref',
    '--prefix',
    'graph_prefix',
    required=False,
    type=str,
    default='pre_',
    help='Prefix added to all names in a graph'
)


opt_combined_graph_path = click.option(
    '-cgp',
    '--combined-graph-path',
    'combined_graph_path',
    required=True,
    type=str,
    callback=check_args,
    help='Output combined onnx graph path'
)


opt_io_map = click.option(
    '-io',
    '--io-map',
    'io_map',
    required=True,
    type=str,
    help='Pairs of output/inputs representing outputs of the first graph and inputs of the second graph to be connected'
)

opt_start = click.option(
    '-snn',
    '--start-node-names',
    'start_node_names',
    required=True,
    type=click.STRING,
    callback=check_node_name,
    help='The names of start nodes'
)


opt_end = click.option(
    '-enn',
    '--end-node-names',
    'end_node_names',
    required=True,
    type=click.STRING,
    callback=check_node_name,
    help='The names of end nodes'
)


opt_check = click.option(
    '-ck',
    '--is-check-subgraph',
    'is_check_subgraph',
    is_flag=True,
    default=False,
    help='Whether to check subgraph. Default to False.'
)


opt_path = click.option(
    '--path',
    'path',
    nargs=1,
    required=True,
    type=str,
    callback=check_args,
    help='Target onnx file or directory containing onnx file'
)


opt_device = click.option(
    '-d',
    '--device',
    'device',
    default=0,
    type=click.IntRange(min=0),
    help='Device_id, default to 0.'
)


opt_loop = click.option(
    '--loop',
    'loop',
    default=100,
    type=click.IntRange(min=1),
    help='How many times to run the test inference, default to 100.'
)


opt_soc = click.option(
    '-soc',
    '--soc-version',
    'soc',
    default='Ascend310P3',
    type=str,
    callback=check_args,
    help='Soc_version, default to Ascend310P3.'
)


opt_converter = click.option(
    '--converter',
    'converter',
    default='atc',
    type=str,
    callback=validate_opt_converter,
    help='OM Converter, default to atc.'
)


opt_threshold = click.option(
    '--threshold',
    'threshold',
    default=0,
    type=click.FloatRange(min=-1),
    help='Threshold of inference speed improvement,'
         'knowledges with less improvement won\'t be used.'
         'Can be a negative number, which means accept'
         'negative optimization, default: 0'
)


opt_infer_test = click.option(
    '-t',
    '--infer-test',
    'infer_test',
    is_flag=True,
    default=False,
    help='Run inference to determine whether to apply knowledges optimization. Default to False.'
)


opt_big_kernel = click.option(
    '-bk',
    '--big-kernel',
    'big_kernel',
    is_flag=True,
    default=False,
    help='Whether to apply big kernel optimize knowledge. Default to False.'
)


opt_attention_start_node = click.option(
    '-as',
    '--attention-start-node',
    'attention_start_node',
    type=str,
    default="",
    callback=check_node_name,
    help='Start node of the first attention block, it must be set when apply big kernel knowledge.',
)


opt_attention_end_node = click.option(
    '-ae',
    '--attention-end-node',
    'attention_end_node',
    type=str,
    default="",
    callback=check_node_name,
    help='End node of the first attention block, it must be set when apply big kernel knowledge.',
)


opt_input_shape = click.option(
    '-is',
    '--input-shape',
    'input_shape',
    type=str,
    callback=check_args,
    help='Input shape of onnx graph.',
)


opt_input_shape_range = click.option(
    '--input-shape-range',
    'input_shape_range',
    type=str,
    callback=check_args,
    help='Specify input shape range for OM converter.'
)


opt_dynamic_shape = click.option(
    '--dynamic-shape',
    'dynamic_shape',
    type=str,
    callback=check_args,
    help='Specify input shape for dynamic onnx in inference.'
)


opt_output_size = click.option(
    '-outsize',
    '--output-size',
    'output_size',
    type=str,
    callback=check_args,
    help='Specify real size of graph output.'
)


opt_subgraph_input_shape = click.option(
    '-sis',
    '--subgraph-input-shape',
    'subgraph_input_shape',
    type=str,
    callback=check_args,
    help='Specify the input shape of subgraph'
)


opt_subgraph_input_dtype = click.option(
    '-sit',
    '--subgraph-input-dtype',
    'subgraph_input_dtype',
    type=str,
    callback=check_args,
    help='Specify the input dtype of subgraph'
)
