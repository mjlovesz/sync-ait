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
from multiprocessing import Pool
import pathlib
from functools import partial
from typing import List

import click
from click_aliases import ClickAliasedGroup
from click.exceptions import UsageError

from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer, InferTestConfig
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern import KnowledgeFactory
from auto_optimizer.tools.log import logger
from auto_optimizer.common.utils import check_output_model_path

from auto_optimizer.options import (
    arg_path,
    arg_input,
    arg_output,
    arg_start,
    arg_end,
    opt_check,
    opt_optimizer,
    opt_recursive,
    opt_verbose,
    opt_soc,
    opt_device,
    opt_infer_test,
    opt_loop,
    opt_threshold,
    opt_input_shape,
    opt_input_shape_range,
    opt_dynamic_shape,
    opt_output_size,
    opt_processes,
)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def is_graph_input_static(graph: BaseGraph) -> bool:
    for input_ in graph.inputs:
        for dim in input_.shape:
            try:
                dim = int(dim)
            except ValueError:
                return False
            if dim <= 0:
                return False
    return True


def optimize_onnx(
    optimizer: GraphOptimizer,
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    infer_test: bool,
    config: InferTestConfig,
) -> List[str]:
    '''Optimize a onnx file and save as a new file.'''
    try:
        graph = OnnxGraph.parse(input_model.as_posix(), add_name_suffix=False)
    except Exception as exc:
        logger.warning('%s model parse failed.', input_model.as_posix())
        logger.warning('exception: %s', exc)
        return []

    config.is_static = is_graph_input_static(graph)
    if infer_test:
        if not (config.is_static or (config.input_shape_range and config.dynamic_shape and config.output_size)):
            logger.warning('Failed to optimize %s with inference test.', input_model.as_posix())
            logger.warning('Didn\'t specify input_shape_range or dynamic_shape or output_size.')
            return []
    optimize_action = partial(optimizer.apply_knowledges_with_infer_test, cfg=config) \
        if infer_test else optimizer.apply_knowledges

    try:
        graph_opt, applied_knowledges = optimize_action(graph=graph)
    except Exception as exc:
        logger.warning('%s optimize failed.', input_model.as_posix())
        logger.warning('exception: %s', exc)
        return []
    
    if applied_knowledges:
        if not output_model.parent.exists():
            output_model.parent.mkdir(parents=True)
        graph_opt.save(output_model.as_posix())
    return applied_knowledges


def evaluate_onnx(
    model: pathlib.Path,
    optimizer: GraphOptimizer,
    verbose: bool,
) -> List[str]:
    '''Search knowledge pattern in a onnx model.'''
    if verbose:
        logger.info(f'Evaluating {model.as_posix()}')
    try:
        graph = OnnxGraph.parse(model.as_posix(), add_name_suffix=False)
        graph, applied_knowledges = optimizer.apply_knowledges(graph)
        return applied_knowledges
    except Exception as exc:
        logger.warning('%s match failed.', model.as_posix())
        logger.warning('exception: %s', exc)
        return []


class FormatMsg:
    def show(self, file=None) -> None:
        logger.error(self.format_message())


@click.group(cls=ClickAliasedGroup, context_settings=CONTEXT_SETTINGS, 
             short_help='Modify ONNX models, and auto optimizer onnx models.',
             no_args_is_help=True)
def cli() -> None:
    '''main entrance of auto optimizer.'''
    pass


@cli.command('list', short_help='List available Knowledges.', context_settings=CONTEXT_SETTINGS)
def command_list() -> None:
    registered_knowledges = KnowledgeFactory.get_knowledge_pool()
    logger.info('Available knowledges:')
    for idx, name in enumerate(registered_knowledges):
        logger.info(f'  {idx:2d} {name}')


@cli.command(
    'evaluate',
    aliases=['eva'],
    short_help='Evaluate model matching specified knowledges.',
    context_settings=CONTEXT_SETTINGS
)
@arg_path
@opt_optimizer
@opt_recursive
@opt_verbose
@opt_processes
def command_evaluate(
    path: pathlib.Path,
    optimizer: GraphOptimizer,
    recursive: bool,
    verbose: bool,
    processes: int,
) -> None:
    path_ = pathlib.Path(path.decode()) if isinstance(path, bytes) else path
    onnx_files = list(path_.rglob('*.onnx') if recursive else path_.glob('*.onnx')) \
        if path_.is_dir() else [path_]

    if processes > 1:
        evaluate = partial(evaluate_onnx, optimizer=optimizer, verbose=verbose)
        with Pool(processes) as p:
            res = p.map(evaluate, onnx_files)
        for file, knowledges in zip(onnx_files, res):
            if not knowledges:
                continue
            summary = ','.join(knowledges)
            logger.info(f'{file}\t{summary}')
        return

    for onnx_file in onnx_files:
        knowledges = evaluate_onnx(optimizer=optimizer, model=onnx_file, verbose=verbose)
        if not knowledges:
            continue
        summary = ','.join(knowledges)
        logger.info(f'{onnx_file}\t{summary}')


@cli.command(
    'optimize',
    aliases=['opt'],
    short_help='Optimize model with specified knowledges.',
    context_settings=CONTEXT_SETTINGS
)
@arg_input
@arg_output
@opt_optimizer
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
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    optimizer: GraphOptimizer,
    infer_test: bool,
    soc: str,
    device: int,
    loop: int,
    threshold: float,
    input_shape: str,
    input_shape_range: str,
    dynamic_shape: str,
    output_size: str
) -> None:
    # compatibility for click < 8.0
    input_model_ = pathlib.Path(input_model.decode()) if isinstance(input_model, bytes) else input_model
    output_model_ = pathlib.Path(output_model.decode()) if isinstance(output_model, bytes) else output_model
    if input_model_ == output_model_:
        logger.warning('output_model is input_model, refuse to overwrite origin model!')
        return
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
        optimizer=optimizer,
        input_model=input_model_,
        output_model=output_model_,
        infer_test=infer_test,
        config=config,
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
    context_settings=CONTEXT_SETTINGS
)
@arg_input
@arg_output
@arg_start
@arg_end
@opt_check
def command_extract(
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    start_node_names: str,
    end_node_names: str,
    is_check_subgraph
) -> None:
    if input_model == output_model:
        logger.warning('output_model is input_model, refuse to overwrite origin model!')
        return
    output_model_path = output_model.as_posix()
    if not check_output_model_path(output_model_path):
        return

    # parse start node names and end node names
    start_nodes = [node_name.strip() for node_name in start_node_names.split(',')]
    end_nodes = [node_name.strip() for node_name in end_node_names.split(',')]

    onnx_graph = OnnxGraph.parse(input_model.as_posix())
    try:
        onnx_graph.extract_subgraph(start_nodes, end_nodes, output_model_path, is_check_subgraph)
    except ValueError as err:
        logger.error(err)


if __name__ == "__main__":
    UsageError.show = FormatMsg.show
    cli()
