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
import re
import json
import queue
import threading
import time
import glob
import datetime
import pytz
import pandas as pd
import torch

from ait_llm.common.log import logger


class OpChecker:
    def __init__(self):
        '''
        cases_info结构：
            'op_id': string
            'op_name': string
            'op_param': dict
            'tensor_path': string
        '''
        self.csv_data = {}
        self.cases_info = {}
        self.completed_op_id_queue = queue.Queue()
        self.special_cases = ['KvCacheOperation', 'ReshapeAndCacheOperation', 'SelfAttentionOperation']
        self.base_path = ''
        self.tensor_path = ''
        self.op_path = ''
        self.output_dir = ''
        self.output_path = ''
        self.ids = ''
        self.check_ids_string = []
        self.opname = None
        self.check_patterns = []
        self.precision_type = []
        utc_time = datetime.datetime.now(tz=pytz.utc)
        self.timestamp = utc_time.astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
        self.rerun = False

    @staticmethod
    def third_party_init():
        execution_flag = True

        import ait_llm

        lib_path = os.environ.get("AIT_OPCHECK_LIB_PATH", "")
        if not lib_path:
            lib_path_dir = os.path.dirname(os.path.abspath(ait_llm.__file__))
            lib_path = os.path.join(lib_path_dir, "opcheck", "libopchecker.so")

        if os.path.exists(lib_path):
            try:
                logger.info(lib_path)
                torch.classes.load_library(lib_path)
            except OSError as e:
                logger_text = "Failed to load libopchecker.so, please check. Error: {}".format(e)
                logger.error(logger_text)
                execution_flag = False
        else:
            logger_text = "libopchecker.so not found in {}".format(lib_path)
            logger.error(logger_text)
            execution_flag = False

        return execution_flag

    def get_base_path(self, cur_path):
        dirseg = cur_path.split(os.path.sep)
        if len(dirseg) >= 4 and dirseg[-3] == 'tensors' and dirseg[-4] == 'ait_dump':
            return cur_path
        elif cur_path == os.path.dirname(cur_path):
            return None
        else:
            return self.get_base_path(os.path.dirname(cur_path))

    def check_input_legality(self, input_path):
        ret = False
        base_path = None

        input_path = os.path.realpath(input_path)
        if not os.path.exists(input_path):
            logger_text = f"Input path not found: {input_path}"
            logger.error(logger_text)
            return input_path, base_path, ret
        
        base_path = self.get_base_path(input_path)
        if base_path is None:
            logger_text = f"input path is not in ait_dump tensors directory: {input_path}"
            logger.error(logger_text)                                                        
            return input_path, base_path, ret
        
        ret = True
        return input_path, base_path, ret

    def args_init(self, args):
        import torch_npu

        execution_flag = True
        
        self.tensor_path, self.base_path, ret = self.check_input_legality(args.input)
        if not ret:
            execution_flag = False
        
        self.output_dir = os.path.realpath(args.output) 
        self.output_path = os.path.join(self.output_dir, f"opcheck_result_{self.timestamp}.xlsx")
        self.ids = args.ids
        if self.ids != '':
            try:
                self.check_ids_string = [x.lower().strip() for x in self.ids.split(',')]
            except ValueError as e:
                logger_text = "Failed to parse ids. Error: {}".format(e)
                logger.error(logger_text)
                execution_flag = False
        self.opname = args.opname
        if self.opname is not None:
            try:
                self.check_patterns = [x.lower().strip() for x in self.opname.split(',')]
            except ValueError as e:
                logger_text = "Failed to parse opname. Error: {}".format(e)
                logger.error(logger_text)
                execution_flag = False
        self.precision_type = args.metric

        # 指定需要使用的npu设备
        try:
            torch.npu.set_device(torch.device(f"npu:{args.device_id}"))
        except RuntimeError as e:
            logger_text = "Failed to set the device. Device_id: {}".format(args.device_id)
            logger.error(logger_text)
            execution_flag = False

        self.rerun = args.rerun
        if self.rerun:
            execution_flag_res = OpChecker.third_party_init()
            if not execution_flag_res:
                execution_flag = False
            else:
                logger_text = "Rerunning operations in atb to calculate outputs..."
                logger.info(logger_text)
        else:
            logger_text = "Comparing outputs in dump data without rerunning operations in atb..."
            logger.info(logger_text)
        return execution_flag

    def start_test(self, args):
        # 0.初始化
        execution_flag_res = self.args_init(args)
        if not execution_flag_res:
            return

        from ait_llm.opcheck.case_manager import CaseManager
        case_manager = CaseManager(self.completed_op_id_queue)
        
        # 1.遍历tensor_path，将算子信息添加到self.cases_info
        self.walk_tensor_path(self.tensor_path)
        logger_text = f"Total {len(self.cases_info)} cases found under path: {self.tensor_path}"
        logger.info(logger_text)

        # 2.将self.cases_info中的用例添加到case_manager
        result_info = 'excuted_information'
        for _, case_info in self.cases_info.items():
            if_successed_add_case = case_manager.add_case(case_info)
            if if_successed_add_case:
                case_info[result_info] = 'addition successed'
            else:
                case_info[result_info] = 'addition failed'

        # 3.执行测试用例并提供专家建议
        self.excute_cases(case_manager)

        # 4.写入未添加成功的算子
        for v in self.cases_info.values():
            if v[result_info] == 'addition failed':
                v['res_detail'] = []
                self.write_op_result_to_csv(v)

    def parse_op_id_name(self, dirpath):
        basename = os.path.basename(dirpath)
        try:
            op_name = basename.split('_')[-1]
        except IndexError as e:
            logger_text = f"{dirpath} is not a valid tensor dir, please check"
            logger.debug(logger_text)
            op_name = None

        rel_path = os.path.relpath(dirpath, self.base_path)
        try:
            op_id = '_'.join([x.split('_')[0] for x in rel_path.split(os.path.sep)])
        except IndexError as e:
            logger_text = f"{dirpath} is not a valid tensor dir, please check"
            logger.debug(logger_text)
            op_id = None

        return op_id, op_name

    def check_id_range(self, op_id):
        if op_id is None:
            return False
        if self.ids == '':
            return True
        else:
            for p in self.check_ids_string:
                ret = re.match("^" + p + "(_[0-9]+){0,20}$", op_id)
                if ret:
                    return True
            return False

    def check_name(self, op_name):
        if op_name is None:
            return False
        if self.opname is None:
            return True
        else:  # 应该是LinearOps，SelfAttention
            for p in self.check_patterns:
                if p in op_name.lower():
                    return True
            return False

    def is_exec_node(self, case_info):
        if self.ids == '' and self.opname is None:
            return True

        flag1 = self.check_id_range(case_info.get("op_id", None))
        flag2 = self.check_name(case_info.get("op_name", None))
        
        if flag1 and flag2:
            return True

        return False

    def add_case_to_cases(self, case_info):
        op_name = case_info.get("op_name", None)
        op_id = case_info.get("op_id", None)
        if op_name == 'KvCacheOperation':
            case_info['inplace_idx'] = [2]
            self.cases_info[op_id] = case_info
        elif op_name == 'ReshapeAndCacheOperation':
            case_info['inplace_idx'] = [2, 3]
            self.cases_info[op_id] = case_info
        elif op_name == 'SelfAttentionOperation':
            self.cases_info[op_id] = case_info
        else:
            self.cases_info[op_id] = case_info

    def add_op_info_to_cases_info(self, dirpath):
        tensor_path = os.path.join(dirpath, 'after')

        json_path = os.path.join(dirpath, 'op_param.json')
        try:
            with open(json_path, 'r') as f:
                op_param = json.load(f)
        except Exception as e:
            logger_text = f"Cannot loads json file to json! Json file: {json_path} \n Exception: {e}"
            logger.debug(logger_text)
            return
            
        op_id, op_name = self.parse_op_id_name(dirpath)
        if op_id is None or op_name is None:
            return

        case_info = {
            'op_id': op_id, 'op_name': op_name, 'op_param': op_param, 'tensor_path': tensor_path, 
            'precision_type': self.precision_type, 'rerun': self.rerun
        }

        ret = self.is_exec_node(case_info)
        if ret:
            self.add_case_to_cases(case_info)
        return

    def walk_tensor_path(self, cur_path):
        files_and_dirs = os.listdir(cur_path)
        dirnames, filenames = [], []
        for item in files_and_dirs:
            item_path = os.path.join(cur_path, item)
            if os.path.isdir(item_path):
                dirnames.append(item)
            else:
                filenames.append(item)
        if 'after' in dirnames and 'op_param.json' in filenames:
            self.add_op_info_to_cases_info(cur_path)
        for dirname in dirnames:
            if dirname != 'after':
                self.walk_tensor_path(os.path.join(cur_path, dirname))

    def excute_cases(self, case_manager):
        # 定义监控队列函数
        def watching_queue():
            cases_num = len([1 for v in self.cases_info.values() if v["excuted_information"] == 'addition successed'])
            cases_index = 0
            while cases_index < cases_num:
                time.sleep(0.1)
                if not self.completed_op_id_queue.empty():
                    completed_op_id = self.completed_op_id_queue.get()
                    case_info = self.cases_info.get(completed_op_id, '')
                    if case_info != '':
                        self.write_op_result_to_csv(case_info)
                    cases_index += 1
                    logger_text = f"===============excuted cases:{cases_index}, total cases:{cases_num}================"
                    logger.info(logger_text)

        watching_thread = threading.Thread(target=watching_queue)
        watching_thread.start()
        case_manager.excute_cases()
        watching_thread.join()

    def get_optional_idx(self):
        optional_idx = []
        if 'abs' in self.precision_type:
            optional_idx.append(0)
            optional_idx.append(1)
        if 'cos_sim' in self.precision_type:
            optional_idx.append(2)
        if 'kl' in self.precision_type:
            optional_idx.append(3)
        return optional_idx

    def write_op_result_to_csv(self, op_result):
        import openpyxl

        optional_idx = self.get_optional_idx()
        if not os.path.exists(self.output_path):
            wb = openpyxl.Workbook()
            ws = wb.active
            required_head = [
                'op_id', 'op_name', 'op_param', 'tensor_path', 'out_tensor_id', 'precision_standard',
                'excuted_information', 'precision_result(%)', 'max_rel_error'
            ]
            optional_head = ['abs_precision_result(%)', 'max_abs_error', 'cosine_similarity', 'kl_divergence']
            optional_head_cp = [optional_head[i] for i in optional_idx]
            ws.append(required_head + optional_head_cp)
            wb.save(self.output_path)

        wb = openpyxl.load_workbook(self.output_path)
        ws = wb.active

        op_id = op_result['op_id']
        op_name = op_result['op_name']
        op_param = json.dumps(op_result['op_param'])
        tensor_path = op_result['tensor_path']
        excuted_information = op_result['excuted_information']
        if len(op_result['res_detail']) > 0:
            for i, res_detail in enumerate(op_result['res_detail']):
                precision_standard = res_detail['precision_standard']
                rel_pass_rate = res_detail['rel_pass_rate']
                max_rel = res_detail['max_rel']
                abs_pass_rate = res_detail['abs_pass_rate']
                max_abs = res_detail['max_abs']
                cos_sim = res_detail['cos_sim']
                kl_div = res_detail['kl_div']
                required = [
                    op_id, op_name, op_param, tensor_path, i, precision_standard, excuted_information, rel_pass_rate,
                    max_rel
                ]
                optional = [abs_pass_rate, max_abs, cos_sim, kl_div]
                optional_cp = [optional[idx] for idx in optional_idx]
                ws.append(required + optional_cp)
        else:
            default_str = 'NaN'
            i, precision_standard, rel_pass_rate, max_rel, abs_pass_rate, max_abs, cos_sim, kl_div = default_str, \
                default_str, default_str, default_str, default_str, default_str, default_str, default_str
            required = [
                op_id, op_name, op_param, tensor_path, i, precision_standard, excuted_information, rel_pass_rate,
                max_rel
            ]
            optional = [abs_pass_rate, max_abs, cos_sim, kl_div]
            optional_cp = [optional[idx] for idx in optional_idx]
            ws.append(required + optional_cp)
        wb.save(self.output_path)