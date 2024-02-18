import os
import shutil

import torch
from torch import nn

from llm import DumpConfig
from llm import register_hook


MODEL_NAME_LIST = ["root", "root.ln"]


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def forward(self, x):
        y = self.ln(x)
        z = y + y
        return z


def test_hook_when_tp_default_then_save_inputs():
    model = SampleModel()
    dump_config = DumpConfig(dump_path="./ait_dump")
    register_hook(model, dump_config)
    x = torch.randn(4, 4)
    model(x)
    for name in MODEL_NAME_LIST:
        except_input_path = os.path.join("./ait_dump", str(os.getpid()) + "_cpu", "0", name, "input_exec1_0.pth")
        except_output_path = os.path.join("./ait_dump", str(os.getpid()) + "_cpu", "0", name, "output_exec1.pth")
        assert os.path.exists(except_input_path)
        assert os.path.exists(except_output_path)
    topo_path = os.path.join("./ait_dump", str(os.getpid()) + "_cpu", "model_tree.json")
    assert os.path.exists(topo_path)
        
    if os.path.exists("./ait_dump"):
        shutil.rmtree("./ait_dump")
