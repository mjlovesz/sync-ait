import numpy
import torch
import torch_npu

from llm.common.log import logger

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []

        self.__parse_bin_file()

    def get_tensor(self):
        if self.dtype == 0:
            dtype = numpy.float32
        elif self.dtype == 1:
            dtype = numpy.float16
        elif self.dtype == 2:  # int8
            dtype = numpy.int8
        elif self.dtype == 3:  # int32
            dtype = numpy.int32
        elif self.dtype == 9:  # int64
            dtype = numpy.int64
        elif self.dtype == 12:
            dtype = numpy.bool8
        else:
            logger_text = f"error, unsupport dtype: {self.dtype}"
            logger.info(logger_text)
            pass
        tensor = torch.tensor(numpy.frombuffer(self.obj_buffer, dtype=dtype))
        tensor = tensor.view(self.dims)
        return tensor
    
    def __parse_bin_file(self):
        end_str = f"{ATTR_END}=1"
        with open(self.file_path, "rb") as fd:
            file_data = fd.read()

            begin_offset = 0
            file_line = [i for i, data in enumerate(file_data) if data == ord("\n")]
            for i in file_line:
                line = file_data[begin_offset: i].decode("utf-8")
                begin_offset = i + 1
                fields = line.split("=")
                attr_name = fields[0]
                attr_value = fields[1]
                if attr_name == ATTR_END:
                    self.obj_buffer = file_data[i + 1:]
                    break
                elif attr_name.startswith("$"):
                    self.__parse_system_atrr(attr_name, attr_value)
                else:
                    self.__parse_user_attr(attr_name, attr_value)
                    pass
                    
    def __parse_system_atrr(self, attr_name, attr_value):
        if attr_name == ATTR_OBJECT_LENGTH:
            self.obj_len = int(attr_value)
        elif attr_name == ATTR_OBJECT_PREFIX:
            pass

    def __parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = int(attr_value)
        elif attr_name == "format":
            self.format = int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            for i, dim in enumerate(self.dims):
                self.dims[i] = int(dim)


def read_tensor(file_path):
    if file_path.endswith(".bin"):
        bin_file = TensorBinFile(file_path)
        return bin_file.get_tensor()
    else:
        try:
            return list(torch.load(file_path).state_dict().values())[0]
        except TypeError as e:
            return torch.load(file_path)