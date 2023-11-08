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
from ais_bench.infer.interface import MultiDeviceSession


def multidevice_infer_pipeline_static():
    device_id = 0
    model_path = "../../sampledata/add_model/model/add_model_bs1.om"
    # create multidevice session of om model for inference
    multi_session = MultiDeviceSession(model_path)
    # create new numpy data
    shape1 = [1,3,32,32]
    shape2 = [1,3,32,32]
    ndata1 = np.full(shape1, 0).astype(np.float32)
    ndata2 = np.full(shape2, 0).astype(np.float32)
    feeds = [ndata1, ndata2]
    feeds_list = [feeds, feeds]
    # create {device_id : input datas} dict
    device_feeds = {device_id:[feeds_list, feeds_list]}
    # in is numpy list and output is numpy list
    outputs = multi_session.infer_pipeline(device_feeds, mode='static')
    print(f"outputs: {outputs}")

multidevice_infer_pipeline_static()