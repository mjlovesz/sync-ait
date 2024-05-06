from array import array

import torch
from torch import float16, float32, int8, int32, int64, bfloat16

ATTR_END = "$END"
ATTR_OBJECT_LENGTH = "$Object.Length"


def read_atb_data(path: str) -> torch.Tensor:
    dtype = 0
    dims = []
    dtype_dict = {
        0: float32,
        1: float16,
        2: int8,
        3: int32,
        9: int64,
        12: torch.bool,
        27: bfloat16
    }

    with open(path, "rb") as fd:
        file_data = fd.read()

    offset = 0
    obj_buffer = ()

    for i, byte in enumerate(file_data):
        if byte == ord("\n"):
            line = file_data[offset: i].decode("utf-8")
            offset = i + 1
            [attr_name, attr_value] = line.split("=")

            if attr_name == ATTR_END:
                obj_buffer = file_data[i + 1:]
                break
            elif attr_name.startswith("$"):
                pass
            else:
                if attr_name == "dtype":
                    dtype = int(attr_value)
                elif attr_name == "dims":
                    dims = [int(x) for x in attr_value.split(",")]

    if dtype not in dtype_dict:
        raise ValueError(f"Unsupported dtype: {dtype}")

    dtype = dtype_dict.get(dtype)
    tensor = torch.frombuffer(array("b", obj_buffer), dtype=dtype)

    return tensor.view(dims)
