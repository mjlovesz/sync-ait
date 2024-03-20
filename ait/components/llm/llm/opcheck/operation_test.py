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
import sys
import re
import unittest
import json
import numpy as np
import torch
import torch_npu

from ait_llm.common.tool import read_atb_data
from ait_llm.common.log import logger
from ait_llm.compare.cmp_algorithm import CMP_ALG_MAP


class OperationTest(unittest.TestCase):
    def __init__(self, methodName='opTest', case_info=None, excuted_ids=None):
        super(OperationTest, self).__init__(methodName)

        self.case_info = case_info
        self.case_info['res_detail'] = []
        self.excuted_ids = excuted_ids
        self.op_id = case_info['op_id']
        self.op_name = case_info['op_name']
        self.op_param = case_info['op_param']
        self.tensor_path = case_info['tensor_path']
        self.in_tensors = []
        self.out_tensors = []
        self.out_dtype = self.case_info["out_dtype"]
        self.rerun = self.case_info["rerun"]
        
        error1 = 'Error0.1‰'
        error2 = 'Error0.5‰'
        error3 = 'Error1‰'
        error4 = 'Error4‰'
        error5 = 'Error5‰'
        error6 = 'Error+/-1'

        self.precision_standard = {
            'ACL_DOUBLE': [error1, 99.99], 'ACL_UINT32': [error1, 99.99], 'ACL_INT64': [error1, 99.99], 
            'ACL_FLOAT': [error1, 99.99], 'ACL_INT32': [error1, 99.99], 'ACL_UINT64': [error1, 99.99], 
            'ACL_FLOAT16': [error3, 99.9], 'ACL_BF16': [error4, 99.6], 'ACL_INT8': [error6, 99.9], 
            'ACL_UINT8': [error6, 99], 'ACL_INT16': [error6, 99.9], 'ACL_UINT16': [error6, 99.9], 
            'ACL_BOOL': [error1, 100], 'double': [error1, 99.99], 'uint32': [error1, 99.99], 
            'int64': [error1, 99.99], 'float': [error1, 99.99], 'int32': [error1, 99.99], 
            'uint64': [error1, 99.99], 'float16': [error3, 99.9], 'bf16': [error4, 99.6], 
            'int8': [error6, 99.9], 'uint8': [error6, 99], 'int16': [error6, 99.9], 
            'uint16': [error6, 99.9], 'bool': [error1, 100]
        }

        self.erol_dict = {
            error1: 0.0001,
            error2: 0.0005,
            error3: 0.001,
            error4: 0.004,
            error5: 0.005,
            error6: 1
        }

    @staticmethod
    def parametrize(optest_class, case_info=None, excuted_ids=None):
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(optest_class)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(optest_class(name, case_info=case_info, excuted_ids=excuted_ids))
        return suite
    
    def setUp(self):
        def get_tensor_path(tensor_type):
            _tensor_path = [x for x in os.listdir(self.tensor_path) if x.startswith(tensor_type)]
            _tensor_path.sort(key=lambda x:int(x.split(tensor_type)[1].split('.')[0]))  
            _tensor_path = [os.path.join(self.tensor_path, x) for x in _tensor_path]
            return _tensor_path

        if self.tensor_path:
            if os.path.isdir(self.tensor_path):
                _in_tensor_path = get_tensor_path("intensor")
                for path in _in_tensor_path:
                    _in_tensor = read_atb_data(path).npu()
                    self.in_tensors.append(_in_tensor)
                _out_tensor_path = get_tensor_path("outtensor")
                for path in _out_tensor_path:
                    _out_tensor = read_atb_data(path).npu()
                    self.out_tensors.append(_out_tensor)
            else:
                raise RuntimeError(f"{self.tensor_path} not valid")
        else:
            raise RuntimeError(f"{self.tensor_path} not valid")
    
    def tearDown(self):
        self.excuted_ids.put(self.op_id)
        if self.case_info['excuted_information'] != 'execution successful':
            self.case_info['excuted_information'] = 'execution failed'
    
    def rerun_op(self, excute_type): 
        operation = torch.classes.OperationTorch.OperationTorch(self.op_name)
        if isinstance(self.op_param, dict):
            operation.set_param(json.dumps(self.op_param))
        elif isinstance(self.op_param, str):
            operation.set_param(self.op_param)
        if excute_type == "inplace":
            operation.execute(self.in_tensors)
            out_tensors = []
            for index in self.case_info['inplace_idx']:
                out_tensors.append(self.in_tensors[index])
        elif excute_type == "with_param":
            operation.set_varaintpack_param(self.case_info['run_param'])
            out_tensors = operation.execute(self.in_tensors)
        else:
            out_tensors = operation.execute(self.in_tensors)
        return out_tensors

    def excute_common(self, excute_type):
        logger_text = f"———————— {self.op_id} {self.op_name} test start ————————"
        logger.info(logger_text)
        if self.rerun:
            out_tensors = self.rerun_op(excute_type)
        else:
            out_tensors = self.out_tensors
        golden_out_tensors = self.golden_calc(self.in_tensors)
        try:
            logger.debug("out_tensor", out_tensors[0].size())
            logger.debug("golden_out_tensor", golden_out_tensors[0].size())
        except TypeError as e:
            logger_text = "The output is abnormal. Please check! Exception: {}".format(e)
            logger.debug(logger_text)

        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def execute(self):
        self.excute_common("common")

    def execute_with_param(self):
        self.excute_common("with_param")

    def execute_inplace(self):
        self.excute_common("inplace")

    def get_rel_pass_rate(self, out, golden, etol):
        out, golden = out.reshape(-1), golden.reshape(-1)
        size = out.shape[0]
        golden_denom = golden.clone().float()
        golden_denom[golden_denom == 0] += torch.finfo(torch.bfloat16).eps
        try:
            rel_errors = torch.abs((out - golden) / golden_denom)
            rel_pass_rate = torch.sum(rel_errors <= etol) / size
        except ZeroDivisionError as e:
            logger_text = "Pass rate of rel error cannot be calculated because the denom is 0. Exception: {}".format(e)
            logger.debug(logger_text)
            raise e
        return rel_pass_rate
    
    def get_max_rel_error(self, out, golden):
        out, golden = out.reshape(-1).float().cpu(), golden.reshape(-1).float().cpu()
        max_rel_error, _ = CMP_ALG_MAP["max_relative_error"](golden, out)
        return max_rel_error

    def get_abs_pass_rate(self, out, golden, etol):
        size = out.shape[0]
        abs_errors = torch.abs(out - golden)
        max_abs_error = torch.max(abs_errors)
        try:
            abs_pass_rate = torch.sum(abs_errors <= etol) / size if size != 0 else 0
        except ZeroDivisionError as e:
            logger_text = "Pass rate of abs error cannot be calculated because the denom is 0. Exception: {}".format(e)
            logger.debug(logger_text)
            abs_pass_rate = None
        return abs_pass_rate, max_abs_error

    def get_cos_similarity(self, out, golden):
        cos_sim, _ = CMP_ALG_MAP["cosine_similarity"](golden, out)
        return cos_sim      
    
    def get_kl_divergence(self, out, golden):
        out, golden = out.tolist(), golden.tolist()
        try:
            out_prob = out / np.sum(out)
            golden_prob = golden / np.sum(golden)
            kl = np.sum(np.where(out_prob != 0, out_prob * np.log(out_prob / golden_prob), 0))
            kl = kl if kl > 0 else 0
        except ZeroDivisionError as e:
            logger_text = "Kl divergence cannot be calculated because the denom is 0. Exception: {}".format(e)
            logger.debug(logger_text)
            kl = None
        return kl
    
    def get_other_precisions(self, out, golden, etol):
        precision_type = self.case_info['precision_type']
        default_str = 'NaN'
        abs_pass_rate, max_abs_error, kl_div = None, None, None
        cos_sim_str = default_str
        
        out, golden = out.reshape(-1).float(), golden.reshape(-1).float()
        if 'abs' in precision_type:
            abs_pass_rate, max_abs_error = self.get_abs_pass_rate(out, golden, etol)
        if 'cos_sim' in precision_type:
            cos_sim_str = self.get_cos_similarity(out, golden)
        if 'kl' in precision_type:
            kl_div = self.get_kl_divergence(out, golden)
        abs_pass_rate_str = "%.16f" % float(abs_pass_rate.item() * 100) if abs_pass_rate is not None else default_str
        max_abs_error_str = "%.16f" % float(max_abs_error.item()) if max_abs_error is not None else default_str
        kl_div_str = "%.16f" % kl_div if kl_div is not None else default_str

        return abs_pass_rate_str, max_abs_error_str, cos_sim_str, kl_div_str
        
    def get_npu_device(self):
        npu_device = os.environ.get("NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device

    def get_soc_version(self):
        device_name = torch.npu.get_device_name()
        if re.search("Ascend910B", device_name, re.I):
            soc_version = 'Ascend910B'
        elif re.search("Ascend310P", device_name, re.I):
            soc_version = 'Ascend310P'
        else:
            raise RuntimeError(f"{device_name} is not supported")
        device_count = torch.npu.device_count()
        current_device = torch.npu.current_device()
        logger_text = "Device Properties: device_name: {}, soc_version: {}, device_count: {}, current_device: {}"\
                    .format(device_name, soc_version, device_count, current_device)
        logger.debug(logger_text)
        return soc_version

    def __golden_compare_all(self, out_tensors, golden_out_tensors):
        flag = True

        try:
            self.assertEqual(len(out_tensors), len(golden_out_tensors))
            self.assertEqual(len(out_tensors), len(self.out_dtype))
        except AssertionError as e:
            flag = False
            logger.debug(e)

        tensor_count = len(out_tensors)
        for i in range(tensor_count):
            p_s = self.precision_standard.get(self.out_dtype[i], [])
            if len(p_s) != 2:
                raise RuntimeError(f"{self.out_dtype[i]} not supported!")
            etol = self.erol_dict.get(p_s[0], 0.001)
            err_rate = p_s[1]
            ps_standard = f"{err_rate}%(error<{etol})"

            rel_pass_rate = self.get_rel_pass_rate(out_tensors[i], golden_out_tensors[i], etol)
            max_rel = self.get_max_rel_error(out_tensors[i], golden_out_tensors[i])

            try:
                self.assertLess(err_rate, rel_pass_rate * 100)
            except AssertionError as e:
                flag = False
                logger.debug(e)
            
            rel_pass_rate = "%.16f" % float(rel_pass_rate.item() * 100)
            max_rel = "%.16f" % float(max_rel)
            abs_pass_rate, max_abs, cos_sim, kl_div = self.get_other_precisions(out_tensors[i], golden_out_tensors[i], 
                                                                                etol)

            self.case_info['res_detail'].append({"precision_standard": ps_standard,
                                                "rel_pass_rate": rel_pass_rate,
                                                "max_rel": max_rel,
                                                "abs_pass_rate": abs_pass_rate,
                                                "max_abs": max_abs,
                                                "cos_sim": cos_sim,
                                                "kl_div": kl_div})
            
            if flag:
                self.case_info['excuted_information'] = 'execution successful'
            else:
                self.case_info['excuted_information'] = 'execution failed'