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

import torch
import torch_npu

from ait_llm.opcheck import operation_test
from ait_llm.common.log import logger


class ActivationGolden:
    @staticmethod
    def relu_golden(in_tensors, _):
        return torch.nn.functional.relu(in_tensors)

    @staticmethod
    def gelu_golden(in_tensors, _):
        try:
            float_result = 0.5 * in_tensors * (1 + torch.nn.functional.tanh(torch.sqrt(2 / torch.pi) * 
                                                    (in_tensors + 0.044715 * torch.pow(in_tensors, 3))))
        except ZeroDivisionError as e:
            raise e
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result

    @staticmethod
    def fast_gelu_golden(in_tensors, _):
        try:
            float_result = in_tensors * torch.exp(0.851 * (in_tensors - torch.abs(in_tensors))) / (1 + 
                            torch.exp(-1.702 * torch.abs(in_tensors)))
        except ZeroDivisionError as e:
            raise e
        return float_result.half()

    @staticmethod
    def swish_golden(in_tensors, scale):
        try:
            float_result = in_tensors / (1 + torch.exp(-in_tensors * scale))
        except ZeroDivisionError as e:
            raise e
        return float_result.half()

    @staticmethod
    def log_golden(in_tensors, _):
        float_result = torch.log(in_tensors)
        return float_result.half() if in_tensors.dtype == torch.float16 else float_result


class OpcheckActivationOperation(operation_test.OperationTest):
    golden_func = {
        1: ActivationGolden.relu_golden,
        2: ActivationGolden.gelu_golden,
        3: ActivationGolden.fast_gelu_golden,
        4: ActivationGolden.swish_golden,
        5: ActivationGolden.log_golden
    } 

    def golden_calc(self, in_tensors):
        activation_type = self.op_param.get("activationType", None)
        scale = self.op_param.get("scale", None)         
        golden_result = OpcheckActivationOperation.golden_func[activation_type](in_tensors[0], scale)
        return [golden_result]

    def test(self):
        activation_type = self.op_param.get("activationType", None)
        scale = self.op_param.get("scale", None)
        if not activation_type or not scale:
            msg = "Cannot get golden data because opParam is not correctly set!"
            logger.error(msg)
            return
        self.execute()