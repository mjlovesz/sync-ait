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

from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory


def convert_to_graph_optimizer(ctx: click.Context, param: click.Option, value: str) -> GraphOptimizer:
    '''Process and validate knowledges option.'''
    try:
        return GraphOptimizer([v.strip() for v in value.split(',')])
    except Exception as err:
        raise click.BadParameter('No valid knowledge provided!') from err


def check_args(ctx: click.Context, params: click.Option, value: str):
    """
    check whether the param is provided
    """
    args = [
        opt
        for param in ctx.command.params
        for opt in param.opts
    ]
    if value in args:
        raise click.MissingParameter()
    return value


default_off_knowledges = [
    'KnowledgeEmptySliceFix',
    'KnowledgeTopkFix',
    'KnowledgeGatherToSplit',
    'KnowledgeSplitQKVMatmul',
    'KnowledgeDynamicReshape',
    'KnowledgeResizeModeToNearest'
]


opt_optimizer = click.option(
    '-k',
    '--knowledges',
    'optimizer',
    default=','.join(
        knowledge
        for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
        if knowledge not in default_off_knowledges
    ),
    type=str,
    callback=convert_to_graph_optimizer,
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


arg_output = click.option(
    '-o',
    '--output'
    'output_model',
    nargs=1,
    required=True,
    type=click.Path(path_type=pathlib.Path)
)


arg_input = click.argument(
    'input_model',
    nargs=1,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path
    )
)


arg_start = click.argument(
    'start_node_names',
    type=click.STRING,
)


arg_end = click.argument(
    'end_node_names',
    type=click.STRING,
)


opt_check = click.option(
    '-c',
    '--is-check-subgraph',
    'is_check_subgraph',
    is_flag=True,
    default=False,
    help='Whether to check subgraph. Default to False.'
)


arg_path = click.argument(
    'path',
    nargs=1,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        path_type=pathlib.Path
    )
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
    '-l',
    '--loop',
    'loop',
    default=100,
    type=click.IntRange(min=1),
    help='How many times to run the test inference, default to 100.'
)


opt_soc = click.option(
    '-soc',
    '--soc_version',
    'soc',
    default='Ascend310P3',
    type=str,
    callback=check_args,
    help='Soc_version, default to Ascend310P3.'
)


def validate_opt_converter(ctx: click.Context, param: click.Option, value: str) -> str:
    '''Process and validate knowledges option.'''
    if value.lower() not in ['atc']:
        raise click.BadParameter('Invalid converter.')
    return value.lower()


opt_converter = click.option(
    '-c',
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
    callback=check_args,
    help='Start node of the first attention block, it must be set when apply big kernel knowledge.',
)


opt_attention_end_node = click.option(
    '-ae',
    '--attention-end-node',
    'attention_end_node',
    type=str,
    default="",
    callback=check_args,
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
    '--output-size',
    'output_size',
    type=str,
    callback=check_args,
    help='Specify real size of graph output.'
)


opt_subgraph_input_shape = click.option(
    '-sis',
    '--subgraph_input_shape',
    'subgraph_input_shape',
    type=str,
    callback=check_args,
    help='Specify the input shape of subgraph'
)


opt_subgraph_input_dtype = click.option(
    '-sit',
    '--subgraph_input_dtype',
    'subgraph_input_dtype',
    type=str,
    callback=check_args,
    help='Specify the input dtype of subgraph'
)
