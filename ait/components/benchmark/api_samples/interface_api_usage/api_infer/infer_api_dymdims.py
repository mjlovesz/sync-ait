import numpy as np
from ais_bench.infer.interface import InferSession

def infer_api_dymdims():
    device_id = 0
    model_path = "../../sampledata/add_model/model/add_model_dymdims.om"

    # create session of om model for inference
    session = InferSession(device_id, model_path)

    # create new numpy data according inputs info
    shape0 = [4, 3, 64, 64]
    ndata0 = np.full(shape0, 1).astype(np.float32)
    shape1 = [4, 3, 64, 64]
    ndata1 = np.full(shape1, 1).astype(np.float32)
    feeds = [ndata0, ndata1]

    # execute inference, inputs is ndarray list and outputs is ndarray list
    outputs = session.infer(feeds, mode='dymdims')
    print(f"outputs: {outputs}")

    # free model resource and device context of session
    session.free_device()


infer_api_dymdims()