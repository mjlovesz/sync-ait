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
import subprocess
from typing import List, Tuple

import click
from click_aliases import ClickAliasedGroup
from click.exceptions import UsageError

import argparse
from components.parser.parser import BaseCommand
from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer, InferTestConfig, BigKernelConfig,\
    ARGS_REQUIRED_KNOWLEDGES
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.graph_refactor.onnx.node import OnnxNode
from auto_optimizer.tools.log import logger
from auto_optimizer.common.click_utils import optimize_onnx, CONTEXT_SETTINGS, \
    FormatMsg, list_knowledges, cli_eva, check_input_path, check_output_model_path
from auto_optimizer.common.click_utils import default_off_knowledges
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory

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
    opt_prefix,
    opt_combined_graph_path,
)


def check_soc(value):
    ivalue = int(value)
    pre_cmd = "npu-smi info -l"
    res = subprocess.run(pre_cmd.split(), shell=False, stdout=subprocess.PIPE)

    sum = 0
    for line in res.stdout.decode().split('\n'):
        if "Chip Count" in line:
            chip_count = int(line.split()[-1])
            sum += chip_count
    if ivalue >= sum or ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Please check device id.")
    return ivalue


def check_range(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 64:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Range 1 ~ 64.")
    return ivalue


def check_min_num_1(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Minimum value 1.")
    return ivalue


def check_min_num_2(value):
    ivalue = int(value)
    if ivalue < -1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid value.Minimum value -1.")
    return ivalue


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
    if start_node_names:
        start_node_names = [node_name.strip() for node_name in start_node_names.split(',')]

    if end_node_names:
        end_node_names = [node_name.strip() for node_name in end_node_names.split(',')]

    onnx_graph = OnnxGraph.parse(input_model)
    try:
        onnx_graph.extract_subgraph(
            start_node_names, end_node_names,
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
@opt_prefix
@opt_combined_graph_path
def command_concatenate(
    graph1: str,
    graph2: str,
    io_map: str,
    graph_prefix: str,
    combined_graph_path: str
) -> None:
    if not check_input_path(graph1):
        raise TypeError(f"Invalid graph1: {graph1}")
    if not check_input_path(graph2):
        raise TypeError(f"Invalid graph2: {graph2}")

    if not check_output_model_path(combined_graph_path):
        raise TypeError(f"Invalid output: {combined_graph_path}")

    onnx_graph1 = OnnxGraph.parse(graph1)
    onnx_graph2 = OnnxGraph.parse(graph2)

    # parse io_map
    # out0:in0;out1:in1...
    io_map_list = []
    for pair in io_map.strip().split(";"):
        if not pair:
            continue
        out, inp = pair.strip().split(":")
        io_map_list.append((out, inp))

    try:
        combined_graph = OnnxGraph.concat_graph(
            onnx_graph1, onnx_graph2,
            io_map_list,
            prefix=graph_prefix,
            graph_name=combined_graph_path
        )
    except Exception as err:
        logger.error(err)

    try:
        combined_graph.save(combined_graph_path)
    except Exception as err:
        logger.error(err)

    logger.info(
        f'Concatenate ONNX model: {graph1} and ONNX model: {graph2} completed. '
        f'Combined model saved in {combined_graph_path}'
    )


if __name__ == "__main__":
    UsageError.show = FormatMsg.show
    cli()


class ListCommand(BaseCommand):
    def add_arguments(self, parser):
        pass

    def handle(self, args):
        list_knowledges()


class EvaluateCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--path', required=True, type=str,
                            help='Target onnx file or directory containing onnx file')
        parser.add_argument('-know', '--knowledges',
                            default=','.join(
                                knowledge
                                for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
                                if knowledge not in default_off_knowledges
                            ),
                            type=str,
                            help='Knowledges(index/name) you want to apply. Seperate by comma(,), \
                            Default to all except fix knowledges.')
        parser.add_argument('-r', '--recursive', action="store_true", default=False,
                            help='Process onnx in a folder recursively if any folder provided \
                            as PATH. Default to false.')
        parser.add_argument('-v', '--verbose', action="store_true", default=False,
                            help='Show progress in evaluate mode. Default to false.')
        parser.add_argument('-p', '--processes', default=1, type=check_range, 
                            help='Use multiprocessing in evaluate mode, \
                            determine how many processes should be spawned. Default to 1')

    def handle(self, args):
        if not check_input_path(args.path):
            return

        path_ = pathlib.Path(args.path)
        knowledge_list = [v.strip() for v in args.knowledges.split(',')]
        for know in knowledge_list:
            if know in ARGS_REQUIRED_KNOWLEDGES:
                knowledge_list.remove(know)
                logger.warning("Knowledge {} cannot be evaluate".format(know))

        if not knowledge_list:
            return
        optimizer = GraphOptimizer(knowledge_list)
        cli_eva(path_, optimizer, args.recursive, args.verbose, args.processes)


class OptimizeCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('-in', '--input', dest='input_model', required=True, type=str,
                            help='Input onnx model to be optimized')
        parser.add_argument('-of', '--output-file', dest='output_model', required=True, type=str,
                            help='Output onnx model name')
        parser.add_argument('-know', '--knowledges',
                            default=','.join(
                                knowledge
                                for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
                                if knowledge not in default_off_knowledges
                            ),
                            type=str,
                            help='Knowledges(index/name) you want to apply. Seperate by comma(,), \
                            Default to all except fix knowledges.')
        parser.add_argument('-t', '--infer-test', action="store_true", default=False,
                            help='Run inference to determine whether to apply knowledges \
                            optimization. Default to False.')
        parser.add_argument('-bk', '--big-kernel', action="store_true", default=False,
                            help='Whether to apply big kernel optimize knowledge. Default to False.')
        parser.add_argument('-as', '--attention-start-node', type=str, default="",
                            help='Start node of the first attention block, \
                            it must be set when apply big kernel knowledge.')
        parser.add_argument('-ae', '--attention-end-node', type=str, default="",
                            help='End node of the first attention block, \
                            it must be set when apply big kernel knowledge.',)
        parser.add_argument('-soc', '--soc-version', dest='soc', default='Ascend310P3', type=str,
                            help='Soc_version, default to Ascend310P3.')
        parser.add_argument('-d', '--device', default=0, type=check_soc,
                            help='Device_id, default to 0.')
        parser.add_argument('--loop', default=100, type=check_min_num_1,
                            help='How many times to run the test inference, default to 100.')
        parser.add_argument('--threshold', default=0, type=check_min_num_2,
                            help='Threshold of inference speed improvement,'
                            'knowledges with less improvement won\'t be used.'
                            'Can be a negative number, which means accept'
                            'negative optimization, default: 0')
        parser.add_argument('-is', '--input-shape', type=str,
                            help='Input shape of onnx graph.',)
        parser.add_argument('--input-shape-range', type=str,
                            help='Specify input shape range for OM converter.')
        parser.add_argument('--dynamic-shape', type=str,
                            help='Specify input shape for dynamic onnx in inference.')
        parser.add_argument('-outsize', '--output-size', type=str,
                            help='Specify real size of graph output.')

    def handle(self, args):
        if not check_input_path(args.input_model) or not check_output_model_path(args.output_model):
            return

        # compatibility for click < 8.0
        input_model_ = pathlib.Path(os.path.abspath(args.input_model))
        output_model_ = pathlib.Path(os.path.abspath(args.output_model))
        if input_model_ == output_model_:
            logger.warning('output_model is input_model, refuse to overwrite origin model!')
            return

        knowledge_list = [v.strip() for v in args.knowledges.split(',')]
        for know in knowledge_list:
            if not args.big_kernel and know in ARGS_REQUIRED_KNOWLEDGES:
                knowledge_list.remove(know)
                logger.warning("Knowledge {} cannot be ran when close big_kernel config.".format(know))

        if not knowledge_list:
            return

        if args.big_kernel:
            big_kernel_config = BigKernelConfig(
                attention_start_node=args.attention_start_node,
                attention_end_node=args.attention_end_node
            )
        else:
            big_kernel_config = None

        config = InferTestConfig(
            converter='atc',
            soc=args.soc,
            device=args.device,
            loop=args.loop,
            threshold=args.threshold,
            input_shape=args.input_shape,
            input_shape_range=args.input_shape_range,
            dynamic_shape=args.dynamic_shape,
            output_size=args.output_size,
        )
        applied_knowledges = optimize_onnx(
            optimizer=knowledge_list,
            input_model=input_model_,
            output_model=output_model_,
            infer_test=args.infer_test,
            config=config,
            big_kernel_config=big_kernel_config
        )
        if args.infer_test:
            logger.info('=' * 100)
        if applied_knowledges:
            logger.info('Result: Success')
            logger.info('Applied knowledges: ')
            for knowledge in applied_knowledges:
                logger.info(f'  {knowledge}')
            logger.info(f'Path: {input_model_} -> {output_model_}')
        else:
            logger.info('Result: Unable to optimize, no knowledges matched.')
        if args.infer_test:
            logger.info('=' * 100)

class ExtractCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('-in', '--input', dest='input_model', required=True, type=str,
                            help='Input onnx model to be optimized')
        parser.add_argument('-of', '--output-file', dest='output_model', required=True, type=str,
                            help='Output onnx model name')
        parser.add_argument('-snn', '--start-node-names', required=False, type=str,
                            help='The names of start nodes')
        parser.add_argument('-enn', '--end-node-names', required=False, type=str,
                            help='The names of end nodes')
        parser.add_argument('-ck', '--is-check-subgraph', action="store_true", default=False,
                            help='Whether to check subgraph. Default to False.')
        parser.add_argument('-sis', '--subgraph-input-shape', type=str,
                            help='Specify the input shape of subgraph')
        parser.add_argument('-sit', '--subgraph-input-dtype', type=str,
                            help='Specify the input dtype of subgraph')

    def handle(self, args):
        if not check_input_path(args.input_model) or not check_output_model_path(args.output_model):
            return

        input_model_ = pathlib.Path(os.path.abspath(args.input_model))
        output_model_ = pathlib.Path(os.path.abspath(args.output_model))
        if input_model_ == output_model_:
            logger.warning('output_model is input_model, refuse to overwrite origin model!')
            return

        # parse start node names and end node names
        if args.start_node_names:
            start_node_names = [node_name.strip() for node_name in args.start_node_names.split(',')]

        if args.end_node_names:
            end_node_names = [node_name.strip() for node_name in args.end_node_names.split(',')]

        onnx_graph = OnnxGraph.parse(args.input_model)
        try:
            onnx_graph.extract_subgraph(
                start_node_names, end_node_names,
                args.output_model, args.is_check_subgraph,
                args.subgraph_input_shape, args.subgraph_input_dtype
            )
        except ValueError as err:
            logger.error(err)

class ConcatenateCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('-g1', '--graph1', required=True, type=str,
                            help='First onnx model to be consolidated')
        parser.add_argument('-g2', '--graph2', required=True, type=str,
                            help='Second onnx model to be consolidated')
        parser.add_argument('-io', '--io-map', required=True, type=str,
                            help='Pairs of output/inputs representing outputs \
                            of the first graph and inputs of the second graph to be connected')
        parser.add_argument('-pref', '--prefix', dest='graph_prefix', 
                            required=False, type=str, default='pre_',
                            help='Prefix added to all names in a graph')
        parser.add_argument('-cgp', '--combined-graph-path', required=True, type=str,
                            help='Output combined onnx graph path')

    def handle(self, args):
        if not check_input_path(args.graph1):
            raise TypeError(f"Invalid graph1: {args.graph1}")
        if not check_input_path(args.graph2):
            raise TypeError(f"Invalid graph2: {args.graph2}")

        if not check_output_model_path(args.combined_graph_path):
            raise TypeError(f"Invalid output: {args.combined_graph_path}")

        onnx_graph1 = OnnxGraph.parse(args.graph1)
        onnx_graph2 = OnnxGraph.parse(args.graph2)

        # parse io_map
        # out0:in0;out1:in1...
        io_map_list = []
        for pair in args.io_map.strip().split(";"):
            if not pair:
                continue
            out, inp = pair.strip().split(":")
            io_map_list.append((out, inp))

        try:
            combined_graph = OnnxGraph.concat_graph(
                onnx_graph1, onnx_graph2,
                io_map_list,
                prefix=args.graph_prefix,
                graph_name=args.combined_graph_path
            )
        except Exception as err:
            logger.error(err)

        try:
            combined_graph.save(args.combined_graph_path)
        except Exception as err:
            logger.error(err)

        logger.info(
            f'Concatenate ONNX model: {args.graph1} and ONNX model: {args.graph2} completed. '
            f'Combined model saved in {args.combined_graph_path}'
        )

class SurgeonCommand(BaseCommand):
    def __init__(self, name="", help="", children=None):
        super().__init__(name, help, children)

    def add_arguments(self, parser, **kwargs):
        return super().add_arguments(parser, **kwargs)

    def handle(self, args, **kwargs):
        return super().handle(args, **kwargs)


def get_cmd_instance():
    surgeon_help_info = "surgeon tool for onnx modifying functions."
    list_cmd_instance = ListCommand("list", "List available Knowledges")
    evaluate_cmd_instance = EvaluateCommand("evaluate", "Evaluate model matching specified knowledges")
    optimize_cmd_instance = OptimizeCommand("optimize", "Optimize model with specified knowledges")
    extract_cmd_instance = ExtractCommand("extract", "Extract subgraph from onnx model")
    concatenate_cmd_instance = ConcatenateCommand("concatenate",
                                                  "Concatenate two onnxgraph into combined one onnxgraph")
    return SurgeonCommand("surgeon", surgeon_help_info, [list_cmd_instance, evaluate_cmd_instance, 
                                                         optimize_cmd_instance, extract_cmd_instance,
                                                         concatenate_cmd_instance])