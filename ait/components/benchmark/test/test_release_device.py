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
import sys
import numpy as np

from ais_bench.infer.interface import InferSession

model_path = sys.argv[1]

def infer_sample():
    device_id = 0
    session1 = InferSession(device_id, model_path)
    session2 = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    barray1 = bytearray(session1.get_inputs()[0].realsize)
    ndata1 = np.frombuffer(barray1)
    barray2 = bytearray(session2.get_inputs()[0].realsize)
    ndata2 = np.frombuffer(barray2)

    # in is numpy list and ouput is numpy list
    outputs1 = session1.infer([ndata1])
    outputs2 = session2.infer([ndata2])
    print(f"outputs1:{outputs1} type:{type(outputs1)}")
    print(f"outputs2:{outputs2} type:{type(outputs2)}")

    print(f"static infer avg 1:{np.mean(session1.sumary().exec_time_list)} ms")
    print(f"static infer avg 2:{np.mean(session2.sumary().exec_time_list)} ms")

infer_sample()