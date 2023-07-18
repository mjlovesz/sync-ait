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
import pathlib

from typing import List, Tuple

import click
from click_aliases import ClickAliasedGroup
from click.exceptions import UsageError

from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer, InferTestConfig, BigKernelConfig,\
    ARGS_REQUIRED_KNOWLEDGES
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.tools.log import logger
from auto_optimizer.common.click_utils import optimize_onnx, CONTEXT_SETTINGS, \
    FormatMsg, list_knowledges, cli_eva, check_input_path, check_output_model_path

from auto_optimizer.ait_options import (
    opt_path,
    opt_input,
    opt_output,
    opt_start,
    opt_end,
    opt_check,
    opt_knowledges,
    opt_recursive,
    opt_verbose,
    opt_soc,
    opt_device,
    opt_big_kernel,
    opt_attention_start_node,
    opt_attention_end_node,
    opt_infer_test,
    opt_loop,
    opt_threshold,
    opt_input_shape,
    opt_input_shape_range,
    opt_dynamic_shape,
    opt_output_size,
    opt_processes,
    opt_subgraph_input_shape,
    opt_subgraph_input_dtype,
    opt_graph1,
    opt_graph2,
    opt_io_map,
    opt_combined_graph_path
)


@click.group(cls=ClickAliasedGroup, context_settings=CONTEXT_SETTINGS,
             short_help='Modify ONNX models, and auto optimizer onnx models.',
             no_args_is_help=True)
def cli() -> None:
    '''main entrance of auto optimizer.'''
    pass


@cli.command('list', short_help='List available Knowledges.', context_settings=CONTEXT_SETTINGS)
def command_list() -> None:
    list_knowledges()


@cli.command(
    'evaluate',
    aliases=['eva'],
    short_help='Evaluate model matching specified knowledges.',
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True
)
@opt_path
@opt_knowledges
@opt_recursive
@opt_verbose
@opt_processes
def command_evaluate(
    path: str,
    knowledges: str,
    recursive: bool,
    verbose: bool,
    processes: int,
) -> None:
    if not check_input_path(path):
        return

    path_ = pathlib.Path(path)
    knowledge_list = [v.strip() for v in knowledges.split(',')]
    for know in knowledge_list:
        if know in ARGS_REQUIRED_KNOWLEDGES:
            knowledge_list.remove(know)
            logger.warning("Knowledge {} cannot be evaluate".format(know))

    if not knowledge_list:
        return
    optimizer = GraphOptimizer(knowledge_list)
    cli_eva(path_, optimizer, recursive, verbose, processes)


@cli.command(
    'optimize',
    aliases=['opt'],
    short_help='Optimize model with specified knowledges.',
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True
)
@opt_input
@opt_output
@opt_knowledges
@opt_big_kernel
@opt_attention_start_node
@opt_attention_end_node
@opt_infer_test
@opt_soc
@opt_device
@opt_loop
@opt_threshold
@opt_input_shape
@opt_input_shape_range
@opt_dynamic_shape
@opt_output_size
def command_optimize(
    input_model: str,
    output_model: str,
    knowledges: str,
    infer_test: bool,
    big_kernel: bool,
    attention_start_node: str,
    attention_end_node: str,
    soc: str,
    device: int,
    loop: int,
    threshold: float,
    input_shape: str,
    input_shape_range: str,
    dynamic_shape: str,
    output_size: str
) -> None:
    if not check_input_path(input_model) or not check_output_model_path(output_model):
        return

    # compatibility for click < 8.0
    input_model_ = pathlib.Path(os.path.abspath(input_model))
    output_model_ = pathlib.Path(os.path.abspath(output_model))
    if input_model_ == output_model_:
        logger.warning('output_model is input_model, refuse to overwrite origin model!')
        return

    knowledge_list = [v.strip() for v in knowledges.split(',')]
    for know in knowledge_list:
        if not big_kernel and know in ARGS_REQUIRED_KNOWLEDGES:
            knowledge_list.remove(know)
            logger.warning("Knowledge {} cannot be ran when close big_kernel config.".format(know))

    if not knowledge_list:
        return

    if big_kernel:
        big_kernel_config = BigKernelConfig(
            attention_start_node=attention_start_node,
            attention_end_node=attention_end_node
        )
    else:
        big_kernel_config = None

    config = InferTestConfig(
        converter='atc',
        soc=soc,
        device=device,
        loop=loop,
        threshold=threshold,
        input_shape=input_shape,
        input_shape_range=input_shape_range,
        dynamic_shape=dynamic_shape,
        output_size=output_size,
    )
    applied_knowledges = optimize_onnx(
        optimizer=knowledge_list,
        input_model=input_model_,
        output_model=output_model_,
        infer_test=infer_test,
        config=config,
        big_kernel_config=big_kernel_config
    )
    if infer_test:
        logger.info('=' * 100)
    if applied_knowledges:
        logger.info('Result: Success')
        logger.info('Applied knowledges: ')
        for knowledge in applied_knowledges:
            logger.info(f'  {knowledge}')
        logger.info(f'Path: {input_model_} -> {output_model_}')
    else:
        logger.info('Result: Unable to optimize, no knowledges matched.')
    if infer_test:
        logger.info('=' * 100)


@cli.command(
    'extract',
    aliases=['ext'],
    short_help='Extract subgraph from onnx model.',
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True
)
@opt_input
@opt_output
@opt_start
@opt_end
@opt_check
@opt_subgraph_input_shape
@opt_subgraph_input_dtype
def command_extract(
    input_model: str,
    output_model: str,
    start_node_names: str,
    end_node_names: str,
    is_check_subgraph: bool,
    subgraph_input_shape: str,
    subgraph_input_dtype: str
) -> None:
    if not check_input_path(input_model) or not check_output_model_path(output_model):
        return

    input_model_ = pathlib.Path(os.path.abspath(input_model))
    output_model_ = pathlib.Path(os.path.abspath(output_model))
    if input_model_ == output_model_:
        logger.warning('output_model is input_model, refuse to overwrite origin model!')
        return

    # parse start node names and end node names
    start_nodes = [node_name.strip() for node_name in start_node_names.split(',')]
    end_nodes = [node_name.strip() for node_name in end_node_names.split(',')]

    onnx_graph = OnnxGraph.parse(input_model)
    try:
        onnx_graph.extract_subgraph(
            start_nodes, end_nodes,
            output_model, is_check_subgraph,
            subgraph_input_shape, subgraph_input_dtype
        )
    except ValueError as err:
        logger.error(err)


@cli.command(
    'concatenate',
    aliases=['concat'],
    short_help='Concatenate two graphs into one',
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True
)
@opt_graph1
@opt_graph2
@opt_io_map
@opt_combined_graph_path
def command_concatenate(
    graph1: str,
    graph2: str,
    io_map: str,
    combined_graph_path: str
) -> None:
    if not check_input_path(graph1):
        raise ValueError(f"Invalid graph1: {graph1}")
    if not check_input_path(graph2):
        raise ValueError(f"Invalid graph2: {graph2}")

    if not check_output_model_path(combined_graph_path):
        raise ValueError(f"Invalid output: {combined_graph_path}")

    graph1_model = pathlib.Path(os.path.abspath(graph1))
    graph2_model = pathlib.Path(os.path.abspath(graph2))
    onnx_graph1 = OnnxGraph.parse(graph1_model)
    onnx_graph2 = OnnxGraph.parse(graph2_model)

    # parse io_map
    # out0:in0;out1:in1...
    io_map_list = [
        (elem[0], elem[1])
        for pair in io_map.strip().split(";")
        for elem in pair.strip().split(":")
    ]

    try:
        combined_graph = OnnxGraph.concat_graph(
            onnx_graph1, onnx_graph2,
            io_map_list
        )
        combined_graph.save(combined_graph_path)
    except ValueError as err:
        logger.error(err)


if __name__ == "__main__":
    UsageError.show = FormatMsg.show
    cli()
