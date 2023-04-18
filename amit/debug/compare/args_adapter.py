# Copyright 2022 Huawei Technologies Co., Ltd
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

class MyArgs:
    def __init__(self,
                 model_path,
                 offline_model_path,
                 input_path,
                 cann_path,
                 out_path,
                 input_shape,
                 device,
                 output_size,
                 output_nodes,
                 advisor):
        self.model_path = model_path
        self.offline_model_path = offline_model_path
        self.input_path = input_path
        self.cann_path = cann_path
        self.out_path = out_path
        self.input_shape = input_shape
        self.device = device
        self.output_size = output_size
        self.output_nodes = output_nodes
        self.advisor = advisor