
import torch

from msquickcmp.pta_acl_cmp.compare import set_label


if __name__ == "__main__":
    set_label(data_src="pta", data_id="a1", data_val=torch.Tensor(1))
    set_label(data_src="acl", data_id="a1", data_val=torch.Tensor(1))

    set_label(data_src="acl", data_id="a2", tensor_path="./outtensor.bin")
    set_label(data_src="pta", data_id="a2", data_val=torch.Tensor(2))
