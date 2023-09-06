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
from tqdm import tqdm
import numpy as np

from ais_bench.infer.interface import InferSession

model_path = sys.argv[1]

def infer_loop_create_session(loop_times):
    device_id = 0
    session_list = []
    loop_list = list(range(loop_times))
    for _, _ in enumerate(tqdm(loop_list, file=sys.stdout, desc='constructing new InferSession:')):
        session = InferSession(device_id, model_path)
        barray = bytearray(session.get_inputs()[0].realsize)
        ndata = np.frombuffer(barray)
        outputs = session.infer([ndata])
        session_list.append(session)
        session.finalize()


infer_loop_create_session(100)