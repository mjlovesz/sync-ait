# Copyright (c) 2023 Huawei Technologies Co., Ltd.
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
import csv

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Any

from model_eval.common import logger, utils
from model_eval.common import Const
from model_eval.common.enum import Engine

OP_FILTER_LIST = ['Constant', 'Const', 'Input', 'Placeholder']


@dataclass
class OpResult:
    ''' Operator analysis result '''
    ori_op_name: str = ''
    ori_op_type: str = '' # origin op type
    op_name: str = ''
    op_type: str = '' # inner op type
    op_engine: Engine = Engine.UNKNOWN
    soc_type: str = ''
    is_supported: bool = True
    details: str = ''

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == 'is_supported':
            if not isinstance(__value, bool) or __value:
                return
        if __name == 'op_type':
            if self.op_type != '':
                return
        return super().__setattr__(__name, __value)

    def set_details(self, err_detail: str) -> None:
        if len(self.details) != 0:
            if err_detail not in self.details.split(';'):
                self.details += ';' + err_detail
        else:
            self.details = err_detail


class Result:
    def __init__(self) -> None:
        self._op_results: Dict[str, OpResult] = {}
    
    def insert(self, op_result: OpResult) -> None:
        ori_op = op_result.ori_op_name
        if isinstance(ori_op, str):
            self._op_results[ori_op] = deepcopy(op_result)

    def get(self, ori_op: str) -> OpResult:
        return self._op_results.get(ori_op)

    def dump(self, out_path: str):
        out_csv = os.path.join(out_path, 'result.csv')
        if not utils.check_file_security(out_csv):
            return
        if os.path.isfile(out_csv):
            os.remove(out_csv)
        try:
            f = open(out_csv, 'x', newline='')
        except Exception as e:
            logger.error(f'open result.csv failed, err:{e}')
        fields = [
            'ori_op_name',
            'ori_op_type',
            'op_name',
            'op_type',
            'soc_type',
            'engine',
            'is_supported',
            'details'
        ]
        writer = csv.DictWriter(f, fieldnames = fields)
        writer.writeheader()
        for op_result in self._op_results.values():
            if op_result.ori_op_type in OP_FILTER_LIST:
                continue
            row = {
                'ori_op_name': op_result.ori_op_name,
                'ori_op_type': op_result.ori_op_type,
                'op_name': op_result.op_name,
                'op_type': op_result.op_type,
                'soc_type': op_result.soc_type,
                'engine': op_result.op_engine.name,
                'is_supported': op_result.is_supported,
                'details': op_result.details
            }
            writer.writerow(row)
        f.flush()
        f.close()
        os.chmod(out_csv, Const.ONLY_READ)
        logger.info(f'Analysis result has bean writted in {out_csv}')
