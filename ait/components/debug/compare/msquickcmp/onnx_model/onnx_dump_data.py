#!/usr/bin/env python
# coding=utf-8
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
"""
Function:
This class is used to generate GUP dump data of the ONNX model.
"""
import sys
import time
import os
import re

import onnx
import onnxruntime
import numpy as np
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model

from msquickcmp.common.dump_data import DumpData
from msquickcmp.common import utils
from msquickcmp.common.utils import AccuracyCompareException
from msquickcmp.common.utils import InputShapeError
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.npu.npu_dump_data_bin2npy import data_convert_file

NODE_TYPE_TO_DTYPE_MAP = {
    "tensor(int)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.double,
    "tensor(bool)": np.bool_,
    "tensor(complex64)": np.complex64,
    "tensor(complex128)": np.complex_
}
MAX_PROTOBUF = 2000000000


class OnnxDumpData(DumpData):
    """
    This class is used to generate GUP dump data of the ONNX model.
    """

    def __init__(self, arguments:CmpArgsAdapter):
        super().__init__()
        self.args = arguments
        self.input_shapes = utils.parse_input_shape(self.args.input_shape)
        self.net_output = {}

        self.data_dir = ""
        self.onnx_dump_data_dir = ""
        self.model_dir = ""
        self.old_onnx_model = onnx.load(self.args.model_path)
        self.new_onnx_model_path = ""
        self.inputs_map = {}

        self._create_dir()
        self.onnx_model_before_custom_op_path = ""
        self.new_onnx_model_before_custom_op_path = ""
        self.onnx_model_after_custom_op_path = ""
        self._extract_sub_models_by_custom_op()

    @staticmethod
    def _check_input_shape_fix_value(op_name, model_shape, input_shape):
        message = "fixed input tensor dim not equal to model input dim." \
                  "tensor_name:%s, %s vs %s" % (op_name, str(input_shape), str(model_shape))
        if len(model_shape) != len(input_shape):
            utils.logger.error(message)
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
        for index, value in enumerate(model_shape):
            if value is None or isinstance(value, str):
                continue
            if input_shape[index] != value:
                utils.logger.error(message)
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

    def generate_inputs_data(self, npu_dump_data_path, use_aipp):
        """
        Function Description:
            generate inputs data
        """
        if self.args.custom_op == "":
            self.new_onnx_model_path =  self._modify_model_add_outputs_nodes(self.args.model_path)
            session = self._load_session(self.new_onnx_model_path)
        else:
            new_model_path = self._modify_model_add_outputs_nodes(self.onnx_model_before_custom_op_path)
            self.new_onnx_model_before_custom_op_path = new_model_path
            session = self._load_session(self.new_onnx_model_before_custom_op_path)

        inputs_tensor_info = self._get_inputs_tensor_info(session)
        self.inputs_map = self._get_inputs_data(self.data_dir, inputs_tensor_info, npu_dump_data_path, use_aipp)

    def generate_dump_data(self, npu_dump_path, om_parser):
        """
        Function description:
            generate onnx model dump data
        Parameter:
            none
        Return Value:
            onnx model dump data directory
        Exception Description:
            none
        """
        if self.args.custom_op == "":
            onnx_dump_data_dir = self._generate_onnx_model_dump_data(self.new_onnx_model_path, 
                                                                     self.args.model_path)
        else:
            # 1. dump data before custom op
            onnx_dump_data_dir = self._generate_onnx_model_dump_data(self.new_onnx_model_before_custom_op_path, 
                                                                     self.onnx_model_before_custom_op_path)
            
            # 2. dump data before custom op
            self._gen_after_custom_op_dump_data(npu_dump_path, om_parser)
        return onnx_dump_data_dir

    def get_net_output_info(self):
        """
        get_net_output_info
        """
        return self.net_output

    def _create_dir(self):
        # create input directory
        self.data_dir = os.path.join(self.args.out_path, "input")
        utils.create_directory(self.data_dir)

        # create dump_data/onnx directory
        self.onnx_dump_data_dir = os.path.join(self.args.out_path, "dump_data/onnx")
        utils.create_directory(self.onnx_dump_data_dir)

        # create model directory
        self.model_dir = ""
        if self.args.dym_shape_range:
            model_relative_name = "../model"
        else:
            model_relative_name = "model"
            if self.args.dump:
                self.model_dir = os.path.join(self.args.out_path, model_relative_name)
                utils.create_directory(self.model_dir)

    def _generate_onnx_model_dump_data(self, onnx_model_witch_outputs_path, origin_onnx_model_path):
        """
        Function description:
            generate onnx model dump data
        Parameter:
            none
        Return Value:
            onnx model dump data directory
        Exception Description:
            none
        """
        session = self._load_session(onnx_model_witch_outputs_path)
        net_output_node = self._get_net_output_node(origin_onnx_model_path)
        dump_bins = self._run_model(session, self.inputs_map)
        origin_onnx_model = onnx.load(origin_onnx_model_path)
        self._save_dump_data(dump_bins, self.onnx_dump_data_dir, origin_onnx_model, net_output_node)
        return self.onnx_dump_data_dir
    
    def _modify_model_add_outputs_nodes(self, onnx_model_path):
        old_onnx_model = onnx.load(onnx_model_path)
        utils.logger.info("load model success")
        for index, node in enumerate(old_onnx_model.graph.node):
            if not node.name:
                node.name = node.op_type + "_" + str(index)
        if not self.args.dump:
            old_onnx_model_graph_output = old_onnx_model.graph.output
            outputs_name_list = [output_node.name for output_node in old_onnx_model_graph_output]
            outputs_name = [name for name in enumerate_model_node_outputs(old_onnx_model) if name in outputs_name_list]
        else:
            outputs_name = [name for name in enumerate_model_node_outputs(old_onnx_model)]
        new_onnx_model = select_model_inputs_outputs(old_onnx_model, outputs_name)
        new_onnx_model_path = os.path.join(self.model_dir, "new_" + os.path.basename(onnx_model_path))
        bytes_model = new_onnx_model.SerializeToString()
        save_as_external_data_switch = sys.getsizeof(bytes_model) > MAX_PROTOBUF
        onnx.save_model(new_onnx_model,
                        new_onnx_model_path,
                        save_as_external_data=save_as_external_data_switch,
                        location=self.model_dir if save_as_external_data_switch else None)
        utils.logger.info("modify model outputs success: %s", new_onnx_model_path)
        return new_onnx_model_path

    def _get_inputs_tensor_info(self, session):
        inputs_tensor_info = []
        # 'session' is a class of 'onnxruntime.InferenceSession'
        # 'input' is a class of 'onnxruntime.NodeArg'
        input_tensor_names = [item.name for item in session.get_inputs()]
        for _, tensor_name in enumerate(self.input_shapes):
            utils.check_input_name_in_model(input_tensor_names, tensor_name)
        for input_item in session.get_inputs():
            tensor_name = input_item.name
            tensor_type = input_item.type
            tensor_shape = tuple(input_item.shape)
            if utils.check_dynamic_shape(tensor_shape):
                if not self.input_shapes:
                    utils.logger.error(
                        "The dynamic shape {} are not supported. Please "
                        "set '-is' or '--input-shape' to fix the dynamic shape.".format(tensor_shape))
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            if self.input_shapes and tensor_name in self.input_shapes:
                input_shape = self.input_shapes.get(tensor_name)
                try:
                    number_shape = [int(dim) for dim in input_shape]
                except (ValueError, TypeError) as error:
                    utils.logger.error(utils.get_shape_not_match_message(
                        InputShapeError.FORMAT_NOT_MATCH, self.args.input_shape))
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR) from error
                self._check_input_shape_fix_value(tensor_name, tensor_shape, number_shape)
                tensor_info = {"name": tensor_name, "shape": tuple(number_shape), "type": tensor_type}
                utils.logger.info("Fix dynamic input shape of %s to %s" % (tensor_name, number_shape))
            else:
                tensor_info = {"name": tensor_name, "shape": tensor_shape, "type": tensor_type}
            inputs_tensor_info.append(tensor_info)
        utils.logger.info("model inputs tensor info:\n{}\n".format(inputs_tensor_info))
        return inputs_tensor_info

    def _get_inputs_data(self, data_dir, inputs_tensor_info, npu_dump_data_path, use_aipp):
        inputs_map = {}
        if use_aipp:
            inputs_map = self._get_inputs_data_aipp(data_dir, inputs_tensor_info, npu_dump_data_path)
            return inputs_map
        if "" == self.args.input_path:
            for i, tensor_info in enumerate(inputs_tensor_info):
                input_data = np.random.random(tensor_info["shape"]).astype(
                    self._convert_to_numpy_type(tensor_info["type"]))
                inputs_map[tensor_info["name"]] = input_data
                file_name = "input_" + str(i) + ".bin"
                input_data.tofile(os.path.join(data_dir, file_name))
                utils.logger.info("save input file name: {}, shape: {}, dtype: {}".format(
                    file_name, input_data.shape, input_data.dtype))
            return inputs_map
        input_path = []
        input_initial_path = self.args.input_path.split(",")
        for input_item in input_initial_path:
            input_item_path = os.path.realpath(input_item)
            if input_item_path.endswith('.bin'):
                input_path.append(input_item_path)
            else:
                utils.get_input_path(input_item_path, input_path)
        if len(inputs_tensor_info) != len(input_path):
            utils.logger.error("the number of model inputs tensor_info is not equal the number of "
                                "inputs data, inputs tensor_info is: {}, inputs data is: {}".format(
                len(inputs_tensor_info), len(input_path)))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
        for i, tensor_info in enumerate(inputs_tensor_info):
            input_data = np.fromfile(input_path[i], self._convert_to_numpy_type(tensor_info["type"])).reshape(
                tensor_info["shape"])
            inputs_map[tensor_info["name"]] = input_data
            utils.logger.info("load input file name: {}, shape: {}, dtype: {}".format(
                os.path.basename(input_path[i]), input_data.shape, input_data.dtype))
        return inputs_map
    
    def _get_inputs_data_aipp(self, data_dir, inputs_tensor_info, npu_dump_data_path):
        inputs_map = {}
        aipp_input = []
        if not npu_dump_data_path:
            utils.logger.error("find no aipp op in dump data, please check --dump is True")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
        for bin_file in os.listdir(npu_dump_data_path):
            if bin_file.startswith("Aipp"):
                aipp_input.append(os.path.join(npu_dump_data_path, bin_file))
        for i, tensor_info in enumerate(inputs_tensor_info):
            data_convert_file(aipp_input[i], os.path.join(self.args.out_path, "input"), self.args)
            aipp_output_path = os.path.join(self.args.out_path, "input", aipp_input[i].rsplit("/", 1)[1]) + \
                               ".output.0.npy"
            aipp_output = np.load(aipp_output_path)
            nchw_prod = np.prod(tensor_info["shape"])
            nchwc_prod_without_c1 = np.prod(aipp_output.shape[:-1])
            try:
                c0 = int(nchw_prod / nchwc_prod_without_c1)
            except ZeroDivisionError as e:
                utils.logger.error("Aipp output has wrong shape, file path: {}".format(aipp_output_path))
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR) from e
            onnx_input = aipp_output[..., :c0].transpose((0, 4, 2, 3, 1)).squeeze(-1).astype(np.float32)
            inputs_map[tensor_info["name"]] = onnx_input
        return inputs_map

    def _convert_to_numpy_type(self, tensor_type):
        numpy_data_type = NODE_TYPE_TO_DTYPE_MAP.get(tensor_type)
        if numpy_data_type:
            return numpy_data_type
        else:
            utils.logger.error(
                "unsupported tensor type: {}".format(tensor_type))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_TENSOR_TYPE_ERROR)

    def _load_session(self, new_onnx_model_path):
        options = onnxruntime.SessionOptions()
        if not self.args.onnx_fusion_switch:
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        return onnxruntime.InferenceSession(new_onnx_model_path, options)

    def _run_model(self, session, inputs_map):
        outputs_name = [node.name for node in session.get_outputs()]
        return session.run(outputs_name, inputs_map)

    def _save_dump_data(self, dump_bins, onnx_dump_data_dir, old_onnx_model, net_output_node):
        res_idx = 0
        for node in old_onnx_model.graph.node:
            for j, output in enumerate(node.output):
                if not self.args.dump and output not in net_output_node:
                    continue
                file_name = node.name.replace('.', '_').replace('/', '_') + "." + str(j) + "." \
                            + str(round(time.time() * 1000000)) + ".npy"
                file_path = os.path.join(onnx_dump_data_dir, file_name)
                if output in net_output_node:
                    self.net_output[net_output_node.index(output)] = file_path
                np.save(file_path, dump_bins[res_idx])
                res_idx += 1
        if not self.args.single_op:
            for key, value in self.net_output.items():
                utils.logger.info("net_output node is:{}, file path is {}".format(key, value))
        utils.logger.info("dump data success")

    def _get_net_output_node(self, onnx_model_path):
        """
        get net output name
        """
        net_output_node = []
        session = self._load_session(onnx_model_path)
        for output_item in session.get_outputs():
            net_output_node.append(output_item.name)
        return net_output_node

    def _extract_sub_models_by_custom_op(self):
        if self.args.custom_op == "":
            return
        try:
            from auto_optimizer import OnnxGraph
        except ModuleNotFoundError as err:
            utils.logger.error("auto_optimizer is not install!")
            raise err
        
        old_onnx_graph = OnnxGraph.parse(self.args.model_path)
        old_onnx_graph.infer_shape()
        custom_op_node = old_onnx_graph[self.args.custom_op]
        if custom_op_node is None:
            utils.logger.error("can't find custom op: %s", self.args.custom_op)
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
        
        self._extract_model_before_custom_op(old_onnx_graph, custom_op_node)
        self._extract_model_after_custom_op(old_onnx_graph, custom_op_node)

    def _extract_model_before_custom_op(self, old_onnx_graph, custom_op_node):
        # start from inputs
        start_nodes_name = []
        for onnx_model_input in old_onnx_graph.inputs:
            start_nodes = old_onnx_graph.get_next_nodes(onnx_model_input.name)
            for start_node in start_nodes:
                if start_node is not None:
                    start_nodes_name.append(start_node.name)

        # end before custom op node
        end_nodes_name = []
        for custom_op_input in custom_op_node.inputs:
            end_node = old_onnx_graph.get_prev_node(custom_op_input)
            end_nodes_name.append(end_node.name)
        
        onnx_model_before_custom_op = old_onnx_graph.extract_subgraph(start_nodes_name, end_nodes_name)
        self.onnx_model_before_custom_op_path = os.path.join(
            self.model_dir, "before_custom_op_" + os.path.basename(self.args.model_path))
        onnx_model_before_custom_op.save(self.onnx_model_before_custom_op_path)
        utils.logger.info("extract model before custom op sucessed, save path: %s", 
                          self.onnx_model_before_custom_op_path)
        
    def _extract_model_after_custom_op(self, old_onnx_graph, custom_op_node):

        # start from custom op outputs
        start_nodes_name = []

        for output in custom_op_node.outputs:
            start_nodes = old_onnx_graph.get_next_nodes(output)
            for start_node in start_nodes:
                if start_node is not None:
                    start_nodes_name.append(start_node.name)
                    utils.logger.info("start_node.name: %s", 
                                    start_node.name)

        # end by old onnx graph outputs
        end_nodes_name = []
        for graph_output in old_onnx_graph.outputs:
            end_node = old_onnx_graph.get_prev_node(graph_output.name)
            end_nodes_name.append(end_node.name)

        onnx_model_after_custom_op = old_onnx_graph.extract_subgraph(start_nodes_name, end_nodes_name)
        self.onnx_model_after_custom_op_path = os.path.join(
            self.model_dir, "after_custom_op_" + os.path.basename(self.args.model_path))
        onnx_model_after_custom_op.save(self.onnx_model_after_custom_op_path)
        utils.logger.info("extract model after custom op sucessed, save path: %s", 
                          self.onnx_model_after_custom_op_path)
        
    def _gen_after_custom_op_dump_data(self, npu_dump_path, om_parser):
        try:
            from auto_optimizer import OnnxGraph
        except ModuleNotFoundError as err:
            utils.logger.error("auto_optimizer is not install!")
            raise err
        
        inputs_tensor_info = self._get_after_custom_op_inputs_ternsor_info()
        inputs_map, inputs_tensor_info = self._get_npu_dump_data_by_custom_op(
            npu_dump_path, inputs_tensor_info, om_parser)
        # fix inputs info 
        onnx_model_after_custom_op = OnnxGraph.parse(self.onnx_model_after_custom_op_path)

        def _update_sub_model_input(onnx_model, input_tensor_info):
            for model_input in onnx_model.inputs:
                if model_input.name == input_tensor_info['name']:
                    model_input.shape = input_tensor_info['shape']
                    model_input.dtype = input_tensor_info['type']

        for input_tensor_info in inputs_tensor_info:
            _update_sub_model_input(onnx_model_after_custom_op, input_tensor_info)

        onnx_model_after_custom_op.infer_shape()
        onnx_model_after_custom_op.save(self.onnx_model_after_custom_op_path)

        # gen dump data
        new_model_path = self._modify_model_add_outputs_nodes(self.onnx_model_after_custom_op_path)        
        session = self._load_session(new_model_path)
        net_output_node = self._get_net_output_node(self.onnx_model_after_custom_op_path)
        dump_bins = self._run_model(session, inputs_map)
        self._save_dump_data(dump_bins, self.onnx_dump_data_dir, 
                             onnx.load(self.onnx_model_after_custom_op_path), 
                             net_output_node)

    def _get_npu_dump_data_by_custom_op(self, npu_dump_path, inputs_tensor_info, om_parser):
        inputs_map = {}

        # 动态bs和动态dim场景，om中的op都会加上_ascend_mbatch_batch_后缀，需要转化下才能匹配上
        custom_op_name = utils.get_mbatch_op_name(om_parser, self.args.custom_op, npu_dump_path)
        utils.logger.info("custom_op_name in npu model:%s", custom_op_name)

        for item in os.listdir(npu_dump_path):
            # file name format: [Optype].[OpName].{time}.[dump_type].[index].npy
            file_name_info = item.split('.')
            op_name = file_name_info[1]
            dump_type = file_name_info[-3]
            index = int(file_name_info[-2])
            if op_name == custom_op_name and dump_type == "output" and index < len(inputs_tensor_info):
                numpy_data = np.load(os.path.join(npu_dump_path, item))
                inputs_tensor_info[index]['shape'] = numpy_data.shape
                inputs_tensor_info[index]['type'] = numpy_data.dtype
                inputs_map[inputs_tensor_info[index]['name']] = numpy_data

        utils.logger.info("extract model after custom op inputs tensor info:\n{}\n".format(inputs_tensor_info))
        return inputs_map, inputs_tensor_info

    def _get_after_custom_op_inputs_ternsor_info(self):
        inputs_tensor_info = []

        session = self._load_session(self.onnx_model_after_custom_op_path)
        for input_item in session.get_inputs():
            tensor_name = input_item.name
            tensor_shape = input_item.shape
            tensor_type = input_item.type

            tensor_info = {"name": tensor_name, "shape": tensor_shape, "type": tensor_type}
            inputs_tensor_info.append(tensor_info)
        return inputs_tensor_info

