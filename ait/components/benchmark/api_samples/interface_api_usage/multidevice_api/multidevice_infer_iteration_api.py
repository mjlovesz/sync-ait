import numpy as np
from ais_bench.infer.interface import MultiDeviceSession


def multidevice_infer_iteration_static():
    device_id = 0
    model_path = "../../sampledata/add_model/model/add_model_bs1.om"
    iteration_times = 1000
    in_out_list = [-1, 0]
    # create multidevice session of om model for inference
    multi_session = MultiDeviceSession(model_path)
    # create new numpy data
    shape1 = [1,3,32,32]
    shape2 = [1,3,32,32]
    ndata1 = np.full(shape1, 0).astype(np.float32)
    ndata2 = np.full(shape2, 0).astype(np.float32)
    # create {device_id : input datas} dict
    device_feeds = {device_id:[[ndata1, ndata2],[ndata1, ndata2]]}
    # in is numpy list and output is numpy list
    outputs = multi_session.infer_iteration(device_feeds, in_out_list, iteration_times, mode='static')


multidevice_infer_iteration_static()