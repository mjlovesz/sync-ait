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

import uuid
import json

from msquickcmp.npu.fusion_parer.fusion_op import OpAttr, OutputDesc, FusionOp
from msquickcmp.common import utils

class ConstManager:
    # fusion rule const
    GRAPH_OBJECT = "graph"
    OP_OBJECT = "op"
    NAME_OBJECT = "name"
    TYPE_OBJECT = "type"
    ID_OBJECT = 'id'
    INPUT_OBJECT = "input"
    ATTR_OBJECT = "attr"
    L1_FUSION_SUB_GRAPH_NO_OBJECT = "_L1_fusion_sub_graph_no"
    ORIGINAL_OP_NAMES_OBJECT = "_datadump_original_op_names"
    OUTPUT_DESC_OBJECT = "output_desc"
    ORIGIN_NAME_OBJECT = "_datadump_origin_name"
    ORIGIN_OUTPUT_INDEX_OBJECT = "_datadump_origin_output_index"
    ORIGIN_FORMAT_OBJECT = "_datadump_origin_format"
    IS_MULTI_OP = "_datadump_is_multiop"
    GE_ORIGIN_FORMAT_OBJECT = "origin_format"
    GE_ORIGIN_SHAPE_OBJECT = "origin_shape"
    D_TYPE = "dtype"
    KEY_OBJECT = "key"
    VALUE_OBJECT = "value"
    STRING_TYPE_OBJECT = "s"
    INT_TYPE_OBJECT = "i"
    BOOL_TYPE_OBJECT = 'b'
    LIST_TYPE_OBJECT = "list"
    DATA_OBJECT = "Data"


class FusionRuleParser:
    """
    the class for parse fusion rule.
    """

    def __init__(self: any, path: str) -> None:
        self.json_path = path
        self.json_object = None
        self.fusion_op_name_to_op_map = {}
        self.op_name_to_fusion_op_name_map = {}
        self.op_list = []
        self.input_nodes = []

    @staticmethod
    def _load_json_file(json_file_path):
        """
        Function Description:
            load json file
        Parameter:
            json_file_path: json file path
        Return Value:
            json object
        Exception Description:
            when invalid json file path throw exception
        """
        try:
            with open(json_file_path, "r") as input_file:
                try:
                    return json.load(input_file)
                except Exception as exc:
                    utils.logger.error('Load Json {} failed, {}'.format(
                        json_file_path, str(exc)))
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR) from exc
        except IOError as input_file_open_except:
            utils.logger.error('Failed to open"' + json_file_path + '", ' + str(input_file_open_except))
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR) from input_file_open_except

    @staticmethod
    def _check_key_exist(json_object: any, key: str) -> None:
        if key not in json_object:
            utils.logger.error('There is no "%s" element in fusion rule file.' % key)
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)
        
    @staticmethod
    def _make_output_desc(output_desc_list: list, name: str) -> None:
        if len(output_desc_list) == 0:
            output_desc = OutputDesc(name, None, "", [])
            output_desc_list.append(output_desc)
        else:
            for (index, _) in enumerate(output_desc_list):
                if output_desc_list[index].origin_name == "":
                    output_desc_list[index].origin_name = name

    def analysis_fusion_rule(self: any) -> None:
        """
        Analysis fusion json file
        """
        self.json_object = self._load_json_file(self.json_path)
        self._parse_fusion_op_json_object()

    def make_fusion_op_name(self: any, name: str, l1_fusion_no: str, original_op_names: list) -> None:
        """
        Make fusion op name by group op name and original op names
        :return the fusion op name
        """
        # the fusion op name priority:
        # l1_fusion_no -> original_op_names -> name
        if l1_fusion_no != "":
            # the l1_fusion_no is not empty,
            # the fusion op name is the l1_fusion_no
            self.op_name_to_fusion_op_name_map[name] = l1_fusion_no
            return

        if original_op_names:
            if len(original_op_names) == 1:
                # There is one original op name
                if original_op_names[0] == '':
                    # the original name is empty, the fusion op name is op name
                    self.op_name_to_fusion_op_name_map[name] = name
                else:
                    # the original name is not empty,
                    # the fusion op name is original op name
                    self.op_name_to_fusion_op_name_map[name] = original_op_names[0]
            else:
                # The original op name more then one,
                # the fusion op name is uuid names
                self.op_name_to_fusion_op_name_map[name] = \
                    uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(original_op_names))
        else:
            self.op_name_to_fusion_op_name_map[name] = name

    def get_origin_name_to_op_name_map(self: any) -> dict:
        """
        Get origin name to op name map
        :return: the map
        """
        origin_name_to_op_name_map = {}
        for fusion_op in self.op_list:
            for origin_name in fusion_op.attr.original_op_names:
                origin_name_to_op_name_map[origin_name] = fusion_op.op_name
        return origin_name_to_op_name_map

    def check_array_object_valid(self: any, json_object: any, key: str) -> None:
        """
        Check array object valid
        :param json_object:the json object
        :param key : key in json
        """
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], list):
            utils.logger.error('The content of the json file "%s" is invalid. The "%s" element is not an array.'
                                % (self.json_path, key))
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)

    def check_string_object_valid(self: any, json_object: any, key: str) -> None:
        """
        Check string object valid
        :param json_object:the json object
        :param key : key in json
        """
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], str):
            utils.logger.error('The content of the json file "%s" is invalid. The "%s" element is not a string.'
                                % (self.json_path, key))
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)

    def get_fusion_op_list(self: any, op_name: str) -> (list, FusionOp):
        """
        Get the fusion op list by op name
        :param op_name: the op name
        :return :the fusion op list, the fusion op by name
        """
        if op_name not in self.op_name_to_fusion_op_name_map:
            message = 'There is no "%s" in the fusion rule file.' % op_name
            utils.logger.error(message)
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)        

        fusion_op_name = self.op_name_to_fusion_op_name_map.get(op_name)
        fusion_op_list = self.fusion_op_name_to_op_map[fusion_op_name]

        # get fusion op in list by op name
        fusion_op_info = None
        for operator in fusion_op_list:
            if operator.op_name == op_name:
                fusion_op_info = operator
                break
        if fusion_op_info is None:
            message = 'There is no "%s" in the fusion rule file.' % op_name
            utils.logger.error(message)
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
        return fusion_op_list, fusion_op_info

    def _adjust_rename_node(self: any) -> None:
        for _, fusion_op_list in self.fusion_op_name_to_op_map.items():
            if len(fusion_op_list) == 1 and self._is_rename_node(fusion_op_list[0]):
                self._make_output_desc(fusion_op_list[0].output_desc,
                                       fusion_op_list[0].attr.original_op_names[0])

    def _parse_fusion_op_json_object(self: any) -> None:
        # check graph element in json file
        self.check_array_object_valid(self.json_object, ConstManager.GRAPH_OBJECT)
        for graph in self.json_object[ConstManager.GRAPH_OBJECT]:
            # check op element in graph value
            self.check_array_object_valid(graph, ConstManager.OP_OBJECT)
            for operator in graph[ConstManager.OP_OBJECT]:
                self._parse_op_object(operator)

            # adjust the output desc for the rename node
            self._adjust_rename_node()

        self.op_list.sort(key=lambda x: x.attr.get_op_sequence())

    def _parse_input_nodes(self, op_object):
        if ConstManager.DATA_OBJECT == op_object.get(ConstManager.TYPE_OBJECT):
            if op_object.get(ConstManager.NAME_OBJECT):
                self.input_nodes.append(op_object.get(ConstManager.NAME_OBJECT))

    def _parse_input(self: any, op_object: any) -> list:
        input_list = []
        # data layer has no input layer
        if ConstManager.INPUT_OBJECT in op_object:
            if not isinstance(op_object[ConstManager.INPUT_OBJECT], list):
                utils.logger.error('The content of the json file "%s" is invalid. The "%s" element is not '
                                    'an array.' % (self.json_path, ConstManager.INPUT_OBJECT))
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)
            for item in op_object[ConstManager.INPUT_OBJECT]:
                if item == "" and len(op_object[ConstManager.INPUT_OBJECT]) == 1:
                    break
                # skip control edge
                if not item.endswith(':-1'):
                    input_list.append(item)
        return input_list

    def _parse_output_desc_in_attr(self: any, output_desc_attr: any, default_index: int) -> OutputDesc:
        origin_name = self._get_string_value_in_attr(output_desc_attr, ConstManager.ORIGIN_NAME_OBJECT)
        origin_output_index = self._get_int_value_in_attr(output_desc_attr, ConstManager.ORIGIN_OUTPUT_INDEX_OBJECT)
        if origin_output_index is None:
            origin_output_index = default_index
        origin_output_format = self._get_string_value_in_attr(output_desc_attr, ConstManager.ORIGIN_FORMAT_OBJECT)
        if origin_output_format == '':
            origin_output_format = self._get_string_value_in_attr(
                output_desc_attr, ConstManager.GE_ORIGIN_FORMAT_OBJECT)
        origin_output_shape = self._get_origin_shape_in_attr(output_desc_attr)
        return OutputDesc(origin_name, origin_output_index, origin_output_format, origin_output_shape)

    def _parse_output_desc(self: any, op_object: any) -> list:
        output_desc_list = []
        # get output desc
        if ConstManager.OUTPUT_DESC_OBJECT in op_object:
            default_index = 0
            for output_desc_object in op_object[ConstManager.OUTPUT_DESC_OBJECT]:

                d_type = ""
                if ConstManager.D_TYPE in output_desc_object:
                    d_type = output_desc_object.get(ConstManager.D_TYPE)

                if ConstManager.ATTR_OBJECT in output_desc_object:
                    output_desc = self._parse_output_desc_in_attr(
                        output_desc_object[ConstManager.ATTR_OBJECT], default_index)
                    output_desc.set_data_type(d_type)
                    output_desc_list.append(output_desc)
                default_index += 1
        return output_desc_list

    def _is_rename_node(self: any, fusion_op: FusionOp) -> bool:
        return len(fusion_op.attr.original_op_names) == 1 and \
               self.op_name_to_fusion_op_name_map.get(fusion_op.op_name) == fusion_op.attr.original_op_names[0]

    def _parse_attr(self: any, op_object: any, op_name: str) -> (OpAttr, bool):
        # check attr element is valid
        if ConstManager.ATTR_OBJECT not in op_object:
            attr_array = []
        else:
            self.check_array_object_valid(op_object, ConstManager.ATTR_OBJECT)
            attr_array = op_object[ConstManager.ATTR_OBJECT]
        is_multi_op = self._get_bool_value_in_attr(attr_array, ConstManager.IS_MULTI_OP)
        # get l1_fusion_sub_graph_no
        l1_fusion_no = self._get_string_value_in_attr(attr_array, ConstManager.L1_FUSION_SUB_GRAPH_NO_OBJECT)
        # get original op names
        original_op_names, have_origin = self._get_original_op_names_in_attr(attr_array, op_name)
        op_sequence = self._parse_id_object(op_object)
        return OpAttr(original_op_names, l1_fusion_no, is_multi_op, op_sequence), have_origin

    def _parse_id_object(self: any, op_object: any) -> int:
        op_sequence = 0
        if ConstManager.ID_OBJECT in op_object:
            self._check_int_object_valid(op_object, ConstManager.ID_OBJECT)
            op_sequence = op_object[ConstManager.ID_OBJECT]
        return op_sequence

    def _parse_op_object(self: any, op_object: any) -> None:
        # check name element is valid
        self.check_string_object_valid(op_object, ConstManager.NAME_OBJECT)
        name = op_object[ConstManager.NAME_OBJECT]
        # check type element is valid
        self.check_string_object_valid(op_object, ConstManager.TYPE_OBJECT)
        self._parse_input_nodes(op_object)

        input_list = self._parse_input(op_object)

        output_desc_list = self._parse_output_desc(op_object)
        attr, have_origin = self._parse_attr(op_object, name)
        if not have_origin:
            self._make_output_desc(output_desc_list, name)

        self.make_fusion_op_name(name, attr.l1_fusion_no, attr.original_op_names)
        fusion_op_name = self.op_name_to_fusion_op_name_map.get(name)
        fusion_op = FusionOp(0, name, input_list, op_object[ConstManager.TYPE_OBJECT], output_desc_list, attr)
        if fusion_op_name in self.fusion_op_name_to_op_map:
            fusion_op.op_id = self.fusion_op_name_to_op_map.get(fusion_op_name)[0].op_id
            self.fusion_op_name_to_op_map.get(fusion_op_name).append(fusion_op)
        else:
            fusion_op.op_id = len(self.fusion_op_name_to_op_map)
            self.fusion_op_name_to_op_map[fusion_op_name] = [fusion_op]
        self.op_list.append(fusion_op)

    def _get_string_value_in_attr(self: any, attr_array: list, key: str) -> str:
        value = ""
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self.check_string_object_valid(value_value, ConstManager.STRING_TYPE_OBJECT)
                value = value_value[ConstManager.STRING_TYPE_OBJECT]
                break
        return value

    def _get_int_value_in_attr(self: any, attr_array: list, key: str) -> int:
        value = None
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self._check_int_object_valid(value_value, ConstManager.INT_TYPE_OBJECT)
                value = value_value[ConstManager.INT_TYPE_OBJECT]
                break
        return value

    def _get_origin_shape_in_attr(self: any, attr_array: list) -> list:
        value = []
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == ConstManager.GE_ORIGIN_SHAPE_OBJECT:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self._check_key_exist(value_value, ConstManager.LIST_TYPE_OBJECT)
                if ConstManager.INT_TYPE_OBJECT in value_value[ConstManager.LIST_TYPE_OBJECT]:
                    self.check_array_object_valid(
                        value_value[ConstManager.LIST_TYPE_OBJECT], ConstManager.INT_TYPE_OBJECT)
                    value = value_value[ConstManager.LIST_TYPE_OBJECT][ConstManager.INT_TYPE_OBJECT]
                break
        return value

    def _get_bool_value_in_attr(self: any, attr_array: list, key: str) -> bool:
        value = False
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self._check_bool_object_valid(value_value, ConstManager.BOOL_TYPE_OBJECT)
                value = value_value[ConstManager.BOOL_TYPE_OBJECT]
                break
        return value

    def _get_original_op_names_in_attr(self: any, attr_array: list, op_name: str) -> (list, bool):
        array = []
        match = False
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == ConstManager.ORIGINAL_OP_NAMES_OBJECT:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value = attr[ConstManager.VALUE_OBJECT]
                self._check_key_exist(value, ConstManager.LIST_TYPE_OBJECT)
                if ConstManager.STRING_TYPE_OBJECT not in value[ConstManager.LIST_TYPE_OBJECT]:
                    array = ['']
                else:
                    self.check_array_object_valid(value[ConstManager.LIST_TYPE_OBJECT],
                                                  ConstManager.STRING_TYPE_OBJECT)
                    array = value[ConstManager.LIST_TYPE_OBJECT][ConstManager.STRING_TYPE_OBJECT]
                match = True
                break
        if not match:
            array.append(op_name)
        return array, match

    def _check_int_object_valid(self: any, json_object: any, key: str) -> None:
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], int):
            utils.logger.error('The content of the json file "%s" is invalid. The "%s" element is not a integer.'
                                % (self.json_path, key))
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)

    def _check_bool_object_valid(self: any, json_object: any, key: str) -> None:
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], bool):
            utils.logger.error('The content of the json file "%s" is invalid. The "%s" element is not a bool.'
                                % (self.json_path, key))
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)
