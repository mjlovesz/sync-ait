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

import tempfile
import warnings
import os
import re
from typing import List, Dict, Union, Sequence, Optional
from collections import deque

import onnx
import numpy as np
from onnx import helper, GraphProto, ModelProto, OperatorSetIdProto, version_converter

from auto_optimizer.graph_refactor import BaseGraph, Initializer, PlaceHolder, Node
from auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder, OnnxInitializer, OnnxNode
from auto_optimizer.tools.log import logger
from auto_optimizer.common.utils import check_output_model_path


class OnnxGraph(BaseGraph):

    def __init__(
        self,
        name: str,
        nodes: Optional[List[OnnxNode]] = None,
        inputs: Optional[List[OnnxPlaceHolder]] = None,
        outputs: Optional[List[OnnxPlaceHolder]] = None,
        initializers: Optional[List[OnnxInitializer]] = None,
        value_infos: Optional[List[OnnxPlaceHolder]] = None,
        **kwargs: Dict[str, object]
    ):
        super(OnnxGraph, self).__init__(name, nodes, inputs, outputs, initializers, value_infos)

        opsets = kwargs.get('opset_imports', 11)
        if isinstance(opsets, int):
            opset_imports = onnx.OperatorSetIdProto()
            opset_imports.version = opsets
            opset_imports = [opset_imports]
        elif isinstance(opsets, Sequence):
            opset_imports = [op for op in opsets if not op.domain or op.domain == '']
            if len(opset_imports) < len(opsets):
                warnings.warn(
                    f'Only one domain version is allowed, keep opset with domain "ai.onnx"')
        else:
            opset_imports = opsets

        self._meta = {
            'ir_version': kwargs.get('ir_version', 4),
            'producer_name': kwargs.get('producer_name', 'AutoOptimizer'),
            'producer_version': kwargs.get('producer_version', 'alpha'),
            'domain': kwargs.get('domain', ''),
            'model_version': kwargs.get('model_version', 0),
            'opset_imports': opset_imports
        }

    @property
    def opset_imports(self) -> Optional[Sequence[OperatorSetIdProto]]:
        return self._meta.get('opset_imports')
    
    @opset_imports.setter
    def opset_imports(self, opset: Union[int, None]) -> None:
        if not opset:
            self._meta['opset_imports'] = None
        else:
            opset_imports = OperatorSetIdProto()
            opset_imports.version = opset
            model = self.model()
            converted_model = version_converter.convert_version(model, opset)
            self.graph = OnnxGraph.parse(converted_model)
            self._meta['opset_imports'] = [opset_imports]

    @classmethod
    def parse(cls, path_or_bytes: Union[str, ModelProto, GraphProto], add_name_suffix: bool = False) -> 'OnnxGraph':
        if isinstance(path_or_bytes, str):
            onnx_model = onnx.load(path_or_bytes)
        if isinstance(path_or_bytes, ModelProto):
            onnx_model = path_or_bytes
        if isinstance(path_or_bytes, GraphProto):
            onnx_graph = path_or_bytes
            meta = {}
        else:
            onnx_graph = onnx_model.graph
            meta = {
                'ir_version': onnx_model.ir_version,
                'domain': onnx_model.domain,
                'model_version': onnx_model.model_version,
                'doc_string': onnx_model.doc_string,
                'opset_imports': onnx_model.opset_import
            }

        inputs = [OnnxPlaceHolder.parse(i) for i in onnx_graph.input]
        outputs = [OnnxPlaceHolder.parse(opt) for opt in onnx_graph.output]
        initializers = [OnnxInitializer.parse(i) for i in onnx_graph.initializer]

        nodes = []
        useless_value_infos = set()
        for node in onnx_graph.node:
            if node.op_type == 'Constant':
                initializers.append(OnnxInitializer.parse(node))
                useless_value_infos.add(node.output[0])
            else:
                nodes.append(OnnxNode.parse(node, add_name_suffix))

        value_infos = []
        for value_info in onnx_graph.value_info:
            if value_info.name not in useless_value_infos:
                value_infos.append(OnnxPlaceHolder.parse(value_info))

        graph = cls(onnx_graph.name, nodes, inputs, outputs, initializers, value_infos, **meta)
        return graph

    def add_input(self, name: str, dtype: str, shape: Sequence[Union[int, str]]) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_input = OnnxPlaceHolder(name, dtype, shape)
        return self._add_input(graph_input)

    def add_output(self, name: str, dtype, shape) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_output = OnnxPlaceHolder(name, dtype, shape)
        return self._add_output(graph_output)

    def add_initializer(self, name: str, value: np.ndarray) -> OnnxInitializer:
        initializer = OnnxInitializer(name, value)
        return self._add_initializer(initializer)

    def add_node(
        self,
        name: str,
        op_type: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, object]] = None,
        domain: str = ''
    ) -> OnnxNode:
        node = OnnxNode(name, op_type, inputs, outputs, attrs=attrs, domain=domain)
        self.update_map()
        return self._add_node(node)

    def proto(self) -> GraphProto:
        self.toposort()
        return helper.make_graph(nodes=[node.proto() for node in self._nodes],
                                 name=self.name,
                                 inputs=[input.proto() for input in self._inputs],
                                 outputs=[output.proto() for output in self._outputs],
                                 initializer=[ini.proto() for ini in self._initializers],
                                 value_info=[val.proto() for val in self._value_infos]
                                 )

    def model(self) -> ModelProto:
        return helper.make_model(self.proto(), **self._meta)

    def save(self, path: str) -> None:
        try:
            onnx.save(self.model(), path)
        except ValueError:
            # large models
            onnx.save(
                self.model(),
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=os.path.basename(path) + '.data'
            )

    def infer_shape(self) -> None:
        # clear value_infos
        self._value_infos = []
        self._value_map = {}
        model = self.model()

        try:
            inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        except ValueError:
            with tempfile.TemporaryDirectory() as tmpdirname:
                onnx.save(
                    model,
                    os.path.join(tmpdirname, 'model.onnx'),
                    save_as_external_data=True
                    )
                onnx.shape_inference.infer_shapes_path(
                    os.path.join(tmpdirname, 'model.onnx'),
                    os.path.join(tmpdirname, 'inferred_model.onnx')
                    )
                inferred_model = onnx.load(os.path.join(tmpdirname, 'inferred_model.onnx'))

       # update value_infos
        graph = inferred_model.graph
        self._value_infos = [OnnxPlaceHolder.parse(v) for v in graph.value_info]
        self._value_map = {v.name: v for v in self._value_infos}

    def extract(
        self,
        new_model_save_path: str,
        input_name_list: List[str],
        output_name_list: List[str],
        enable_model_check: bool = True
    ) -> 'OnnxGraph':

        def check_model(model):
            pass
        if not enable_model_check:
            onnx.checker.check_model = check_model

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.save(os.path.join(tmpdirname, 'model.onnx'))
            logger.info('Begin to extract the model.')
            try:
                onnx.utils.extract_model(
                    os.path.join(tmpdirname, 'model.onnx'),
                    new_model_save_path,
                    input_name_list,
                    output_name_list
                )
            except ValueError as e:
                raise RuntimeError('Function extract() does not support a Large ONNX Model >2GB currently.') from e
            logger.info('Extract the model completed, model saved in {}.'.format(
                new_model_save_path))
        return OnnxGraph.parse(new_model_save_path)

    def extract_subgraph(self,
                         start_node_names: List[str],
                         end_node_names: List[str],
                         subgraph_path: str = None,
                         is_check_subgraph: bool = False,
                         input_shape: str = None,
                         input_dtype: str = None):

        # do shape info by default
        try:
            self.infer_shape()
        except Exception as exp:
            logger.debug("Infer shape failed: %s", exp)

        # parse input info from input shape and input dtype
        input_shape_dict = self._parse_input_info(input_shape)
        input_dtype_dict = self._parse_input_info(input_dtype)

        all_node_names = {node.name for node in self.nodes}
        for start_node_name in start_node_names:
            if start_node_name not in all_node_names:
                raise ValueError(f'Start node {start_node_name} is not in this model')
        for end_node_name in end_node_names:
            if end_node_name not in all_node_names:
                raise ValueError(f'End node {end_node_name} is not in this model')

        input_name_list = []
        for start_node_name in start_node_names:
            start_node = self.get_node(start_node_name, node_type=Node)
            for inp in start_node.inputs:
                if len(inp) == 0:
                    continue
                if not self.get_node(inp, Initializer) and (inp not in input_name_list):
                    input_name_list.append(inp)

        output_name_list = []
        for end_node_name in end_node_names:
            end_node = self.get_node(end_node_name, node_type=Node)
            for oup in end_node.outputs:
                if oup not in output_name_list:
                    output_name_list.append(oup)

        start_nodes = [self.get_node(start_name, node_type=Node) for start_name in start_node_names]
        end_nodes = [self.get_node(end_name, node_type=Node) for end_name in end_node_names]

        top_down_visited = self._bfs_search_reachable_nodes(start_nodes)
        bottom_up_visited = self._bfs_search_reachable_nodes(end_nodes, top_down=False)
        reachable_nodes = top_down_visited & bottom_up_visited

        if not reachable_nodes:
            raise ValueError('There is no path from start nodes to end nodes')

        # collect reachable initializers and value_infos
        initializers = []
        value_infos = []
        for node in reachable_nodes:
            for inp in node.inputs:
                ini = self.get_node(inp, Initializer)
                if ini and ini not in initializers:
                    initializers.append(ini)
                elif self.get_prev_node(inp) not in reachable_nodes and inp not in input_name_list:
                    if len(inp) == 0:
                        continue
                    input_name_list.append(inp)
                elif self.get_node(inp, PlaceHolder) and inp not in input_name_list:
                    value_infos.append(self.get_node(inp, PlaceHolder))

        # check input shape and input dtype
        self._check_input_shape_and_dtype(input_name_list, input_shape_dict, input_dtype_dict)

        # add inputs and outputs for extracted graph
        inputs = self._add_new_io_placeholder(input_name_list, input_shape_dict, input_dtype_dict)
        outputs = self._add_new_io_placeholder(output_name_list)

        # save_model
        subgraph = OnnxGraph('extracted graph', reachable_nodes, inputs, outputs,
                             initializers, value_infos, **self._meta)
        subgraph.toposort()

        if subgraph_path and check_output_model_path(subgraph_path):
            subgraph.save(subgraph_path)
            logger.info('Extract the model completed, model saved in {}.'.format(
                subgraph_path))

        if is_check_subgraph:
            try:
                onnx.checker.check_model(subgraph.model())
            except Exception as exp:
                logger.info("Check subgraph failed, error is:", exp)

        return subgraph
    
    def simplify(self, **kwargs) -> 'OnnxGraph':
        try:
            from onnxsim import simplify
        except ImportError as err:
            raise RuntimeError("No module named 'onnxsim'") from err

        model = self.model()
        model_sim, check = simplify(model, **kwargs)
        if not check:
            raise RuntimeError("Simplified ONNX model could not be validated")

        return OnnxGraph.parse(model_sim)

    def _bfs_search_reachable_nodes(self, start_nodes, top_down=True):
        visited = set()
        queue = deque(start_nodes)
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if top_down:
                for output_name in node.outputs:
                    for next_node in self.get_next_nodes(output_name):
                        queue.append(next_node)
            else:
                for input_name in node.inputs:
                    prev_node = self.get_prev_node(input_name)
                    if prev_node:
                        queue.append(prev_node)
        return visited

    def _add_new_io_placeholder(self, name_list, input_shape_dict=None, input_dtype_dict=None):
        ph_list = []
        for name in name_list:
            value_info = self.get_node(name, PlaceHolder)
            ph_shape = None
            ph_dtype = np.dtype('float32')
            if value_info:
                ph_shape = value_info.shape
                ph_dtype = value_info.dtype
            if input_shape_dict and input_shape_dict.get(name):
                ph_shape = [int(i) for i in input_shape_dict[name]]
            if input_dtype_dict and input_dtype_dict.get(name):
                ph_dtype = np.dtype(input_dtype_dict[name])

            if ph_shape:
                onnx_placeholder = OnnxPlaceHolder(
                    name,
                    ph_dtype,
                    ph_shape
                )
            else:
                onnx_placeholder = OnnxPlaceHolder(
                    name,
                    ph_dtype
                )
            if value_info:
                ph_list.append(onnx_placeholder)
        return ph_list

    def _parse_input_info(self, input_info):
        input_info_dict = {}
        if not input_info:
            return input_info_dict

        input_segs = input_info.strip().split(";")
        for items in input_segs:
            input_field, input_value = items.strip().split(":")
            input_field = input_field.strip()
            input_value = [i.strip() for i in input_value.strip().split(",")]
            input_info_dict[input_field] = input_value

        return input_info_dict

    def _check_input_shape_and_dtype(self,
                                     input_name_list,
                                     input_shape_dict,
                                     input_dtype_dict):
        dtype_converter = {
            'bool': 'bool', 'int': 'int32', 'intc': 'int32',
            'intp': 'int32', 'int8': 'int8', 'int16': 'int16',
            'int32': 'int32', 'int64': 'int64', 'uint8': 'uint8',
            'uint16': 'uint16', 'uint32': 'uint32', 'uint64': 'uint64',
            'float': 'float64', 'float16': 'float16', 'float32': 'float32',
            'float64': 'float64', 'complex': 'complex128', 'complex64': 'complex64',
            'complex128': 'complex128', 'fp16': 'float16', 'fp32': 'float32', 'fp64': 'float64'
        }

        for inp in input_shape_dict.keys():
            if inp not in input_name_list:
                logger.warning(f'Input : {inp} is not in the inputs of the subgraph'
                               f'Please check it or the default shape will be applied.')

        for inp, inp_dtype in input_dtype_dict.items():
            if inp not in input_name_list:
                logger.warning(f'Input : {inp} is not in the inputs of the subgraph'
                               f'Please check it or the default dtype (float32) will be applied.')
            if inp_dtype[0] not in dtype_converter:
                raise ValueError(f"The input type {inp_dtype} of {inp} is not valid. Please check it.")

            input_dtype_dict[inp] = dtype_converter[inp_dtype[0]]


