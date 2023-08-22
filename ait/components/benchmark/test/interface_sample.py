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
import logging
import numpy as np

from ais_bench.infer.interface import InferSession

model_path = sys.argv[1]


# 最短运行样例
def infer_simple():
    device_id = 0
    session = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)

    # in is numpy list and ouput is numpy list
    outputs = session.infer([ndata])
    logging.info(f"outputs:{outputs} type:{type(outputs)}")

    logging.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_loop_inner():
    # only for single_op_add_model
    device_id = 0
    loop_times = 100
    session = InferSession(device_id, model_path, None, False, loop_times)
    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)




def infer_pipeline():
    device_id = 0
    session = InferSession(device_id, model_path)

    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)

    outputs = session.infer([[ndata]])
    print("outputs:{} type:{}".format(outputs, type(outputs)))

    print("static infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))


def infer_torch_tensor():
    import torch
    device_id = 0
    session = InferSession(device_id, model_path)
    # create continuous torch tensor
    torchtensor = torch.zeros([1, 3, 256, 256], out = None, dtype = torch.uint8)
    # in is torch tensor and ouput is numpy list
    outputs = session.infer([torchtensor])
    logging.info(f"in torch tensor outputs[0].shape:{outputs[0].shape} type:{type(outputs)}")

    # create discontinuous torch tensor
    torchtensor = torch.zeros([1, 256, 3, 256], out = None, dtype = torch.uint8)
    torchtensor_discontinue = torchtensor.permute(0, 2, 1, 3)

    # in is discontinuous tensor list and ouput is numpy list
    outputs = session.infer([torchtensor_discontinue])
    logging.info(f"in discontinuous torch tensor outputs[0].shape:{outputs[0].shape} type:{type(outputs)}")

    logging.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_dymshape():
    device_id = 0
    session = InferSession(device_id, model_path)

    ndata = np.zeros([1, 3, 224, 224], dtype = np.float32)

    mode = "dymshape"
    # input args custom_sizes is int
    output_size = 100000
    outputs = session.infer([ndata], mode, custom_sizes=output_size)
    logging.info(f"inputs: custom_sizes: {output_size} outputs:{outputs} type:{type(outputs)}")

    # input args custom_sizes is list
    output_sizes = [100000]
    outputs = session.infer([ndata], mode, custom_sizes=output_sizes)
    logging.info(f"inputs: custom_sizes: {output_sizes} outputs:{outputs} type:{type(outputs)}")
    logging.info(f"dymshape infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_dymdims():
    device_id = 0
    session = InferSession(device_id, model_path)

    ndata = np.zeros([1, 3, 224, 224], dtype = np.float32)

    mode = "dymdims"
    outputs = session.infer([ndata], mode)
    logging.info(f"outputs:{outputs} type:{type(outputs)}")

    logging.info(f"dymdims infer avg:{np.mean(session.sumary().exec_time_list)} ms")


# 获取模型信息
def get_model_info():
    device_id = 0
    session = InferSession(device_id, model_path)

    # 方法2 直接打印session 也可以获取模型信息
    logging.info(session.session)

    # 方法3 也可以直接通过get接口去获取
    intensors_desc = session.get_inputs()
    for i, info in enumerate(intensors_desc):
        logging.info(f"input info i:{i} shape:{info.shape} type:{info.datatype} val: \
                     {int(info.datatype)} realsize:{info.realsize} size:{info.size}")

    intensors_desc = session.get_outputs()
    for i, info in enumerate(intensors_desc):
        logging.info(f"outputs info i:{i} shape:{info.shape} type:{info.datatype} val: \
                     {int(info.datatype)} realsize:{info.realsize} size:{info.size}")

infer_simple()

