# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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
from enum import Enum


class Arg(object):
    def __init__(self, benchmark_arg, atc_arg, msquickcmp_arg):
        self.atc_arg = atc_arg
        self.benchmark_arg = benchmark_arg
        self.msquickcmp_arg = msquickcmp_arg


class DynamicArgumentEnum(Enum):
    # enum struct Arg(benchmark_arg, atc_arg, msquickcmp_arg)
    DYM_BATCH = Arg("--dymBatch", "--dynamic_batch_size", None)
    DYM_SHAPE = Arg("--dymShape", "--input_shape_range", "input_shape")
    DYM_DIMS = Arg("--dymDims", "--dynamic_dims", "input_shape")

    @staticmethod
    def get_all_args() -> list:
        """
        get all argument enum, return as a list
        """
        return list(map(lambda arg: arg, DynamicArgumentEnum))
