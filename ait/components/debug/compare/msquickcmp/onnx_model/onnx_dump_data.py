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
    This class is used to generate dump data of the ONNX model.
    """

    def __init__(self, arguments:CmpArgsAdapter):
        super().__init__()
        self.model_path, self.out_path, self.input_path = arguments.model_path, arguments.out_path, arguments.input_path
        self.input_shape, self.dym_shape_range = arguments.input_shape, arguments.dym_shape_range
        self.custom_op, self.onnx_fusion_switch = arguments.custom_op, arguments.onnx_fusion_switch
        self.dump = arguments.dump
        self.args = arguments

        self._check_path_exists(self.model_path, extentions="onnx")

        self.input_shapes = utils.parse_input_shape(self.input_shape)
        self.data_dir, self.onnx_dump_data_dir, self.model_dir = self._create_dir()

        self.net_output, self.inputs_map = {}, {}
        self.origin_model, self.origin_model_session = self._load_onnx_and_session(self.model_path)

        if self.custom_op:
            head_model, head_model_path, tail_model, tail_model_path = self._extract_sub_models_by_custom_op()
            self.model_before_custom_op, self.model_before_custom_op_path = head_model, head_model_path
            self.model_after_custom_op, self.model_after_custom_op_path = tail_model, tail_model_path

            self.dump_model_with_inputs_path = self._new_model_save_path(self.model_before_custom_op_path)
            self.model_with_inputs = self.model_before_custom_op
            self.model_with_inputs_session = self._load_session(self.model_with_inputs.SerializeToString())
        else:
            self.dump_model_with_inputs_path = self._new_model_save_path(self.model_path)
            self.model_with_inputs, self.model_with_inputs_session = self.origin_model, self.origin_model_session

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

    def generate_inputs_data(self, npu_dump_data_path=None, use_aipp=False):
        inputs_tensor_info = self._get_inputs_tensor_info()
        if use_aipp:
            if not npu_dump_data_path:
                utils.logger.error("find no aipp op in dump data, please check if --dump is True")
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            self._check_path_exists(npu_dump_data_path)
            self.inputs_map = self._get_inputs_data_aipp(self.data_dir, inputs_tensor_info, npu_dump_data_path)
        else:
            self.inputs_map = self._get_inputs_data(self.data_dir, inputs_tensor_info)

    def generate_dump_data(self, npu_dump_path=None, om_parser=None):
        dump_model_with_inputs_contents = self._modify_model_add_outputs_nodes(
            self.model_with_inputs, self.dump_model_with_inputs_path
        )
        session = self._load_session(dump_model_with_inputs_contents)
        dump_bins = self._run_model(session, self.inputs_map)

        net_output_node = [output_item.name for output_item in self.model_with_inputs_session.get_outputs()]
        self._save_dump_data(dump_bins, self.model_with_inputs, net_output_node)

        if self.custom_op:
            # dump data before custom op
            if not npu_dump_path:
                utils.logger.error("npu_dump_path not provided, please check if --dump is True")
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            self._check_path_exists(npu_dump_path)
            self._gen_after_custom_op_dump_data(npu_dump_path, om_parser)
        return self.onnx_dump_data_dir

    def _load_session(self, model_contents):
        options = onnxruntime.SessionOptions()
        if not self.onnx_fusion_switch:
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        return onnxruntime.InferenceSession(model_contents, options)

    def _load_onnx_and_session(self, model_path):
        # model_path str -> read as bytes -> deserialize to onnx_model
        #                                 -> onnxruntime load as session
        with open(model_path, "rb") as ff:
            model_contents = ff.read()
        onnx_model = onnx.load_model_from_string(model_contents)
        for index, node in enumerate(onnx_model.graph.node):
            if not node.name:
                node.name = node.op_type + "_" + str(index)

        onnx_session = self._load_session(model_contents)
        return onnx_model, onnx_session

    def _new_model_save_path(self, origin_path):
        save_name = "new_" + os.path.basename(origin_path)
        return os.path.join(self.model_dir, save_name)

    def _create_dir(self):
        # create input directory
        data_dir = os.path.join(self.out_path, "input")
        utils.create_directory(data_dir)

        # create dump_data/onnx directory
        onnx_dump_data_dir = os.path.join(self.out_path, "dump_data/onnx")
        utils.create_directory(onnx_dump_data_dir)

        # create model directory
        model_dir = ""
        if self.dym_shape_range:
            model_relative_name = "../model"
        else:
            model_relative_name = "model"
            if self.dump:
                model_dir = os.path.join(self.out_path, model_relative_name)
                utils.create_directory(model_dir)
        return data_dir, onnx_dump_data_dir, model_dir

    def _modify_model_add_outputs_nodes(self, onnx_model, save_path):
        if not self.dump:
            origin_model_graph_output = onnx_model.graph.output
            outputs_name_list = [output_node.name for output_node in origin_model_graph_output]
            outputs_name = [name for name in enumerate_model_node_outputs(onnx_model) if name in outputs_name_list]
        else:
            outputs_name = [name for name in enumerate_model_node_outputs(onnx_model)]
        new_onnx_model = select_model_inputs_outputs(onnx_model, outputs_name)
        bytes_model = new_onnx_model.SerializeToString()
        save_as_external_data_switch = sys.getsizeof(bytes_model) > MAX_PROTOBUF
        onnx.save_model(new_onnx_model,
                        save_path,
                        save_as_external_data=save_as_external_data_switch,
                        location=self.model_dir if save_as_external_data_switch else None)
        utils.logger.info("modify model outputs success: %s", save_path)
        return bytes_model

    def _get_inputs_tensor_info(self):
        inputs_tensor_info = []
        input_tensor_names = [item.name for item in self.model_with_inputs_session.get_inputs()]
        for _, tensor_name in enumerate(self.input_shapes):
            utils.check_input_name_in_model(input_tensor_names, tensor_name)
        for input_item in self.model_with_inputs_session.get_inputs():
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
                        InputShapeError.FORMAT_NOT_MATCH, self.input_shape))
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR) from error
                self._check_input_shape_fix_value(tensor_name, tensor_shape, number_shape)
                tensor_info = {"name": tensor_name, "shape": tuple(number_shape), "type": tensor_type}
                utils.logger.info("Fix dynamic input shape of %s to %s" % (tensor_name, number_shape))
            else:
                tensor_info = {"name": tensor_name, "shape": tensor_shape, "type": tensor_type}
            inputs_tensor_info.append(tensor_info)
        utils.logger.info("model inputs tensor info:\n{}\n".format(inputs_tensor_info))
        return inputs_tensor_info

    def _get_inputs_data(self, data_dir, inputs_tensor_info):
        names = [ii["name"] for ii in inputs_tensor_info]
        shapes = [ii["shape"] for ii in inputs_tensor_info]
        dtypes = [self._convert_to_numpy_type(ii["type"]) for ii in inputs_tensor_info]

        if "" == self.input_path:
            return self._generate_random_input_data(data_dir, names, shapes, dtypes)

        input_path = []
        input_initial_path = self.input_path.split(",")
        for input_item in input_initial_path:
            input_item_path = os.path.realpath(input_item)
            if input_item_path.endswith('.bin'):
                input_path.append(input_item_path)
            else:
                utils.get_input_path(input_item_path, input_path)

        self._check_input_data_path(input_path, inputs_tensor_info)
        return self._read_input_data(input_path, names, shapes, dtypes)

    def _get_inputs_data_aipp(self, data_dir, inputs_tensor_info, npu_dump_data_path):
        inputs_map = {}
        aipp_input = []
        for bin_file in os.listdir(npu_dump_data_path):
            if bin_file.startswith("Aipp"):
                aipp_input.append(os.path.join(npu_dump_data_path, bin_file))
        for i, tensor_info in enumerate(inputs_tensor_info):
            data_convert_file(aipp_input[i], os.path.join(self.out_path, "input"), self.args)
            aipp_output_path = os.path.join(self.out_path, "input", aipp_input[i].rsplit("/", 1)[1]) + \
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

    def _run_model(self, session, inputs_map):
        outputs_name = [node.name for node in session.get_outputs()]
        return session.run(outputs_name, inputs_map)

    def _save_dump_data(self, dump_bins, old_onnx_model, net_output_node):
        res_idx = 0
        for node in old_onnx_model.graph.node:
            for j, output in enumerate(node.output):
                if not self.dump and output not in net_output_node:
                    continue
                file_name = self._generate_dump_data_file_name(node.name, j)
                file_path = os.path.join(self.onnx_dump_data_dir, file_name)
                if output in net_output_node:
                    self.net_output[net_output_node.index(output)] = file_path
                np.save(file_path, dump_bins[res_idx])
                res_idx += 1
        for key, value in self.net_output.items():
            utils.logger.info("net_output node is:{}, file path is {}".format(key, value))
        utils.logger.info("dump data success")

    def _extract_sub_models_by_custom_op(self):
        try:
            from auto_optimizer import OnnxGraph
        except ModuleNotFoundError as err:
            utils.logger.error("auto_optimizer is not install!")
            raise err

        origin_model_graph = OnnxGraph.parse(self.origin_model)
        origin_model_graph.infer_shape()
        custom_op_node = origin_model_graph[self.custom_op]
        if custom_op_node is None:
            utils.logger.error("can't find custom op: %s", self.custom_op)
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

        head_model, head_model_path = self._extract_model_before_custom_op(origin_model_graph, custom_op_node)
        tail_model, tail_model_path = self._extract_model_after_custom_op(origin_model_graph, custom_op_node)
        return head_model, head_model_path, tail_model, tail_model_path

    def _extract_model_before_custom_op(self, onnx_graph, custom_op_node):
        # start from inputs
        start_nodes_name = []
        for onnx_model_input in onnx_graph.inputs:
            start_nodes = onnx_graph.get_next_nodes(onnx_model_input.name)
            for start_node in start_nodes:
                if start_node is not None:
                    start_nodes_name.append(start_node.name)

        # end before custom op node
        end_nodes_name = []
        for custom_op_input in custom_op_node.inputs:
            end_node = onnx_graph.get_prev_node(custom_op_input)
            end_nodes_name.append(end_node.name)

        model_before_custom_op = onnx_graph.extract_subgraph(start_nodes_name, end_nodes_name)
        filename = "before_custom_op_" + os.path.basename(self.model_path)
        model_before_custom_op_path = os.path.join(self.model_dir, filename)
        model_before_custom_op.save(model_before_custom_op_path)
        utils.logger.info("extract model before custom op sucessed, save path: %s", model_before_custom_op_path)
        return model_before_custom_op.model(), model_before_custom_op_path

    def _extract_model_after_custom_op(self, onnx_graph, custom_op_node):
        # start from custom op outputs
        start_nodes_name = []
        for output in custom_op_node.outputs:
            start_nodes = onnx_graph.get_next_nodes(output)
            for start_node in start_nodes:
                if start_node is not None:
                    start_nodes_name.append(start_node.name)
                    utils.logger.info("start_node.name: %s", start_node.name)

        # end by old onnx graph outputs
        end_nodes_name = []
        for graph_output in onnx_graph.outputs:
            end_node = onnx_graph.get_prev_node(graph_output.name)
            end_nodes_name.append(end_node.name)

        model_after_custom_op = onnx_graph.extract_subgraph(start_nodes_name, end_nodes_name)
        filename = "after_custom_op_" + os.path.basename(self.model_path)
        model_after_custom_op_path = os.path.join(self.model_dir, filename)
        model_after_custom_op.save(model_after_custom_op_path)
        utils.logger.info("extract model after custom op sucessed, save path: %s", model_after_custom_op_path)
        return model_after_custom_op.model(), model_after_custom_op_path

    def _gen_after_custom_op_dump_data(self, npu_dump_path, om_parser):
        try:
            from auto_optimizer import OnnxGraph
        except ModuleNotFoundError as err:
            utils.logger.error("auto_optimizer is not install!")
            raise err

        model_after_custom_op_session = self._load_session(self.model_after_custom_op.SerializeToString())
        inputs_tensor_info = self._get_after_custom_op_inputs_ternsor_info(model_after_custom_op_session)
        inputs_map, inputs_tensor_info = self._get_npu_dump_data_by_custom_op(
            npu_dump_path, inputs_tensor_info, om_parser)

        # fix inputs info
        model_after_custom_op_graph = OnnxGraph.parse(self.model_after_custom_op)
        inputs_tensor_dict = {ii["name"]: ii for ii in inputs_tensor_info}
        for model_input in model_after_custom_op_graph.inputs:
            input_tensor_info = inputs_tensor_dict.get(model_input.name)
            if input_tensor_info:
                model_input.shape = input_tensor_info['shape']
                model_input.dtype = input_tensor_info['type']

        model_after_custom_op_graph.infer_shape()
        model_after_custom_op_graph.save(self.model_after_custom_op_path)

        # gen dump data
        save_path = self._new_model_save_path(self.model_after_custom_op_path)
        dump_model_with_inputs_contents = self._modify_model_add_outputs_nodes(self.model_after_custom_op, save_path)
        dump_model_with_inputs_session = self._load_session(dump_model_with_inputs_contents)
        dump_bins = self._run_model(dump_model_with_inputs_session, inputs_map)

        net_output_node = [output_item.name for output_item in model_after_custom_op_session.get_outputs()]
        self._save_dump_data(dump_bins, self.model_after_custom_op, net_output_node)

    def _get_npu_dump_data_by_custom_op(self, npu_dump_path, inputs_tensor_info, om_parser):
        inputs_map = {}

        # 动态bs和动态dim场景，om中的op都会加上_ascend_mbatch_batch_后缀，需要转化下才能匹配上
        custom_op_name = utils.get_mbatch_op_name(om_parser, self._to_valid_name(self.custom_op), npu_dump_path)
        utils.logger.info("custom_op_name in npu model:%s", custom_op_name)

        for item in os.listdir(npu_dump_path):
            # file name format: [Optype].[OpName].{time}.[dump_type].[index].npy
            file_name_info = item.split('.')
            if len(file_name_info) < 5:
                continue
            op_name = file_name_info[1]
            dump_type = file_name_info[-3]
            index = int(file_name_info[-2])
            if op_name == custom_op_name and dump_type == "output" and index < len(inputs_tensor_info):
                inputs_info = inputs_tensor_info[index]
                numpy_data = np.load(os.path.join(npu_dump_path, item), allow_pickle=True)
                numpy_data = numpy_data.reshape(inputs_info['shape']).astype(inputs_info['type'])
                # inputs_tensor_info[index]['shape'] = numpy_data.shape
                inputs_tensor_info[index]['type'] = numpy_data.dtype
                inputs_map[inputs_tensor_info[index]['name']] = numpy_data

        if len(inputs_map) != len(inputs_tensor_info):
            required, found = [ii['name'] for ii in inputs_tensor_info], list(inputs_map.keys())
            utils.logger.error(f"Can not find all input_data for custom_op, required: {required}, found: {found}")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

        utils.logger.info("extract model after custom op inputs tensor info:\n{}\n".format(inputs_tensor_info))
        return inputs_map, inputs_tensor_info

    def _get_after_custom_op_inputs_ternsor_info(self, session):
        inputs_tensor_info = []
        for input_item in session.get_inputs():
            numpy_dtype = self._convert_to_numpy_type(input_item.type)
            inputs_tensor_info.append({"name": input_item.name, "shape": input_item.shape, "type": numpy_dtype})
        return inputs_tensor_info
