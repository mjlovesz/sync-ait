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
import time
import logging
import numpy as np

from ais_bench.infer.interface import InferSession, MultiDeviceSession

model_path = sys.argv[1]

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# 最短运行样例
def infer_simple():
    device_id = 0
    session = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)

    # in is numpy list and ouput is numpy list
    outputs = session.infer([ndata])
    logger.info(f"outputs:{outputs} type:{type(outputs)}")

    logger.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_iteration_withD2H():
    # only for single_op_add_model
    device_id = 0
    loop_times = 1  # same infer loop times
    iteration_times = 1000  # inner iteration infer loop times
    session = InferSession(device_id, model_path, None, False, loop_times)
    # create new numpy data according inputs info
    shape = session.get_inputs()[0].shape
    ndata = np.full(shape, 1).astype(np.float32)
    outputs = session.infer([ndata, ndata])
    for i in range(iteration_times - 1):
        outputs = session.infer([outputs[0], ndata])
    logger.info(f"outputs:{outputs} type:{type(outputs)}")
    logger.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_iteration_withoutD2H():
    # only for single_op_add_model
    device_id = 0
    loop_times = 1  # same infer loop times
    in_out_list = [-1, 0]
    iteration_times = 1000  # inner iteration infer loop times
    session = InferSession(device_id, model_path, None, False, loop_times)
    # create new numpy data according inputs info
    shape = session.get_inputs()[0].shape
    ndata = np.full(shape, 1).astype(np.float32)
    outputs = session.infer_iteration([ndata, ndata], in_out_list, iteration_times)
    logger.info(f"outputs:{outputs} type:{type(outputs)}")
    logger.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_dymbatch():
    # only for resnet50 dymbatch
    device_id = 0
    session = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    ndata = np.full([4, 3, 256, 256], 1).astype(np.uint8)

    # in is numpy list and ouput is numpy list
    outputs = session.infer([ndata], "dymbatch")
    logging.info(f"outputs:{outputs} type:{type(outputs)}")
    logging.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_dymhw():
    # only for resnet50 dymbatch
    device_id = 0
    session = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    ndata = np.full([1, 3, 224, 224], 1).astype(np.float32)

    # in is numpy list and ouput is numpy list
    outputs = session.infer([ndata], "dymhw")
    logging.info(f"outputs:{outputs} type:{type(outputs)}")
    logging.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_pipeline():
    device_id = 0
    session = InferSession(device_id, model_path)

    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)

    outputs = session.infer([[ndata], [ndata]])
    print("outputs:{} type:{}".format(outputs, type(outputs)))

    print("static infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))


def infer_multidevices():
    device_id = 0
    multi_session = MultiDeviceSession(device_id, model_path)
    session = InferSession(device_id, model_path)
    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    session.free_resource()
    device_feeds = {0: [[ndata], [ndata]]}
    outputs = multi_session.infer(device_feeds)
    logger.info(f"outputs:{outputs} type:{type(outputs)}")


def infer_multidevices_pipeline():
    device_id = 0
    multi_session = MultiDeviceSession(device_id, model_path)
    session = InferSession(device_id, model_path)
    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    session.free_resource()
    device_feeds_list = {0: [[[ndata], [ndata]], [[ndata], [ndata]]]}
    outputs = multi_session.infer_pipeline(device_feeds_list)
    logger.info(f"outputs:{outputs} type:{type(outputs)}")


def infer_multidevices_iteration():
    # only for single_op_add_model
    device_id = 0
    in_out_list = [-1, 0]
    iteration_times = 1000  # inner iteration infer loop times
    # create new numpy data according inputs info
    multi_session = MultiDeviceSession(device_id, model_path)
    session = InferSession(device_id, model_path)
    # create new numpy data according inputs info
    shape = session.get_inputs()[0].shape
    ndata = np.full(shape, 1).astype(np.float32)
    outputs = session.infer_iteration([ndata, ndata], in_out_list, iteration_times)
    session.free_resource()
    device_feeds = {0: [[ndata, ndata], [ndata, ndata]]}
    outputs = multi_session.infer_iteration(device_feeds, in_out_list, iteration_times)
    logger.info(f"outputs:{outputs} type:{type(outputs)}")


def infer_torch_tensor():
    import torch

    device_id = 0
    session = InferSession(device_id, model_path)
    # create continuous torch tensor
    torchtensor = torch.zeros([1, 3, 256, 256], out=None, dtype=torch.uint8)
    # in is torch tensor and ouput is numpy list
    outputs = session.infer([torchtensor])
    logger.info(f"in torch tensor outputs[0].shape:{outputs[0].shape} type:{type(outputs)}")

    # create discontinuous torch tensor
    torchtensor = torch.zeros([1, 256, 3, 256], out=None, dtype=torch.uint8)
    torchtensor_discontinue = torchtensor.permute(0, 2, 1, 3)

    # in is discontinuous tensor list and ouput is numpy list
    outputs = session.infer([torchtensor_discontinue])
    logger.info(f"in discontinuous torch tensor outputs[0].shape:{outputs[0].shape} type:{type(outputs)}")

    logger.info(f"static infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_dymshape():
    device_id = 0
    session = InferSession(device_id, model_path)

    ndata = np.zeros([1, 3, 224, 224], dtype=np.float32)

    mode = "dymshape"
    # input args custom_sizes is int
    output_size = 100000
    outputs = session.infer([ndata], mode, custom_sizes=output_size)
    logger.info(f"inputs: custom_sizes: {output_size} outputs:{outputs} type:{type(outputs)}")

    # input args custom_sizes is list
    output_sizes = [100000]
    outputs = session.infer([ndata], mode, custom_sizes=output_sizes)
    logger.info(f"inputs: custom_sizes: {output_sizes} outputs:{outputs} type:{type(outputs)}")
    logger.info(f"dymshape infer avg:{np.mean(session.sumary().exec_time_list)} ms")


def infer_dymdims():
    device_id = 0
    session = InferSession(device_id, model_path)

    ndata = np.zeros([1, 3, 224, 224], dtype=np.float32)

    mode = "dymdims"
    outputs = session.infer([ndata], mode)
    logger.info(f"outputs:{outputs} type:{type(outputs)}")

    logger.info(f"dymdims infer avg:{np.mean(session.sumary().exec_time_list)} ms")


# 获取模型信息
def get_model_info():
    device_id = 0
    session = InferSession(device_id, model_path)

    # 方法2 直接打印session 也可以获取模型信息
    logger.info(session.session)

    # 方法3 也可以直接通过get接口去获取
    intensors_desc = session.get_inputs()
    for i, info in enumerate(intensors_desc):
        logger.info(
            f"input info i:{i} shape:{info.shape} type:{info.datatype} val: \
                     {int(info.datatype)} realsize:{info.realsize} size:{info.size}"
        )

    intensors_desc = session.get_outputs()
    for i, info in enumerate(intensors_desc):
        logger.info(
            f"outputs info i:{i} shape:{info.shape} type:{info.datatype} val: \
                     {int(info.datatype)} realsize:{info.realsize} size:{info.size}"
        )


#
start = time.time()
# infer_simple()
# infer_iteration_withD2H()
# infer_multidevices()
# infer_multidevices_iteration()
infer_multidevices_pipeline()
# infer_iteration_withoutD2H()
# infer_dymbatch()
# infer_dymhw()
end = time.time()
e2e_cost = end - start
logger.info(f"endtoend time:{e2e_cost} sec")
