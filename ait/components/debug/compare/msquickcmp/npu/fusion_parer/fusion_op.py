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
FusionOp class. This class mainly involves the fusion op info.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2019-2021
"""
class OutputDesc:
    """
    The class for fusion op output desc
    """

    def __init__(self: any, origin_name: str, origin_output_index: int, origin_format: str,
                 origin_shape: list) -> None:
        self.origin_name = origin_name
        self.origin_output_index = origin_output_index
        self.origin_format = origin_format
        self.origin_shape = origin_shape
        self.data_type = ""

    def get_origin_name(self: any) -> str:
        """
        Get origin name
        """
        return self.origin_name

    def get_origin_shape(self: any) -> list:
        """
        Get origin shape
        """
        return self.origin_shape

    def set_data_type(self: any, data_type: str) -> None:
        """
        Set the data type of output
        """
        self.data_type = data_type


class OpAttr:
    """
    The class for op attr
    """

    def __init__(self: any, original_op_names: list, l1_fusion_no: str, is_multi_op: bool, op_sequence: int) -> None:
        self.original_op_names = original_op_names
        self.l1_fusion_no = l1_fusion_no
        self._is_multi_op = is_multi_op
        self._op_sequence = op_sequence
        self.quant_filter = False

    def is_multi_op(self: any) -> bool:
        """
        is multi op
        """
        return self._is_multi_op

    def get_op_sequence(self: any) -> int:
        """
        Get op sequence
        """
        return self._op_sequence



class FusionOp:
    """
    The class for fusion op
    """

    def __init__(self: any, *args: any) -> None:
        op_id, op_name, input_list, op_type, output_desc, attr = args
        self.op_id = op_id
        self.op_name = op_name
        self.input_list = input_list
        self.op_type = op_type
        self.output_desc = output_desc
        self.attr = attr
