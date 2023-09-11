import numpy as np
from ais_bench.infer.interface import InferSession

def infer_iteration_api_static():
    device_id = 0
    model_path = "../../sampledata/add_model/model/add_model_bs1.om"

    # create session of om model for inference
    session = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    shape0 = session.get_inputs()[0].shape
    ndata0 = np.full(shape0, 1).astype(np.float32)
    shape1 = session.get_inputs()[1].shape
    ndata1 = np.full(shape1, 1).astype(np.float32)
    feeds = [ndata0, ndata1]

    # define unique parameters of infer_iteration
    in_out_list =[-1, 0]
    iteration_times = 100

    # execute inference, inputs is ndarray list and outputs is ndarray list
    outputs = session.infer_iteration(feeds, in_out_list, iteration_times, mode='static')
    print(f"outputs: {outputs}")

    # free model resource and device context of session
    session.free_device()


infer_iteration_api_static()