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

import numpy as np
from ais_bench.infer.interface import InferSession


def infer_api_dymhw():
    device_id = 0
    model_path = "../../sampledata/add_model/model/add_model_dymhw.om"
    # create session of om model for inference
    session = InferSession(device_id, model_path)
    # create new numpy data according inputs info
    shape0 = [1, 3, 32, 32]
    ndata0 = np.full(shape0, 1).astype(np.float32)
    shape1 = [1, 3, 32, 32]
    ndata1 = np.full(shape1, 1).astype(np.float32)
    feeds = [ndata0, ndata1]
    # execute inference, inputs is ndarray list and outputs is ndarray list
    outputs = session.infer(feeds, mode='dymhw')
    print(f"outputs: {outputs}")
    # free model resource and device context of session
    session.free_resource()


infer_api_dymhw()