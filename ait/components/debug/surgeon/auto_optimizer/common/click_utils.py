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
from functools import partial
from typing import List, Optional

import click

from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer, InferTestConfig, BigKernelConfig
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.tools.log import logger


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
    big_kernel_config: Optional[BigKernelConfig] = None
) -> List[str]:
    '''Optimize a onnx file and save as a new file.'''
    try:
        graph = OnnxGraph.parse(input_model.as_posix(), add_name_suffix=False)
    except Exception as exc:
        logger.warning('%s model parse failed.', input_model.as_posix())
        logger.warning('exception: %s', exc)
        return []

    optimizer.init_knowledges()

    if big_kernel_config:
        knowledge_cls = optimizer.knowledges.get("KnowledgeBigKernel")
        if knowledge_cls:
            knowledge_bk = knowledge_cls(
                graph=graph,
                start_node=big_kernel_config.attention_start_node,
                end_node=big_kernel_config.attention_end_node
            )
            optimizer.knowledges.update({"KnowledgeBigKernel": knowledge_bk})
    else:
        optimizer.knowledges.pop("KnowledgeBigKernel")

    config.is_static = is_graph_input_static(graph)
    if infer_test:
        if not (config.is_static or (config.input_shape_range and config.dynamic_shape and config.output_size)):
            logger.warning('Failed to optimize %s with inference test.', input_model.as_posix())
            logger.warning('Didn\'t specify input_shape_range or dynamic_shape or output_size.')
            return []
        optimize_action = partial(optimizer.apply_knowledges_with_infer_test, cfg=config)
    else:
        optimize_action = optimizer.apply_knowledges

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
    except Exception as exc:
        logger.warning('%s match failed.', model.as_posix())
        logger.warning('exception: %s', exc)
        return []
    try:
        graph, applied_knowledges = optimizer.apply_knowledges(graph)
    except Exception as exc:
        logger.warning('%s match failed.', model.as_posix())
        logger.warning('exception: %s', exc)
        return []
    return applied_knowledges


class FormatMsg:
    def show(self, file=None) -> None:
        logger.error(self.format_message())


def convert_to_graph_optimizer(ctx: click.Context, param: click.Option, value: str) -> GraphOptimizer:
    '''Process and validate knowledges option.'''
    try:
        return GraphOptimizer([v.strip() for v in value.split(',')])
    except Exception as err:
        raise click.BadParameter('No valid knowledge provided!') from err


default_off_knowledges = [
    'KnowledgeEmptySliceFix',
    'KnowledgeTopkFix',
    'KnowledgeGatherToSplit',
    'KnowledgeSplitQKVMatmul',
    'KnowledgeDynamicReshape',
    'KnowledgeResizeModeToNearest'
]


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


def validate_opt_converter(ctx: click.Context, param: click.Option, value: str) -> str:
    '''Process and validate knowledges option.'''
    if value.lower() not in ['atc']:
        raise click.BadParameter('Invalid converter.')
    return value.lower()