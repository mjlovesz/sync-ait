from typing import List

import torch.nn as nn

from components.llm.ait_llm.transform.model_parser.kind import mlp, attention, convert


def mname(module: nn.Module):
    return module.__class__.__name__


def add_child(arr, node):
    arr["children"].append(node)


def filter_dropout_module(module: nn.Module):
    ret = []
    children = list(module.children())

    for c in children:
        sub_children = list(c.children())
        if ((len(sub_children) > 0 and isinstance(sub_children[0], nn.Dropout))
                or isinstance(module, nn.Dropout)):
            continue
        ret.append(c)

    return ret


def find_duplicate(modules: List[nn.Module]):
    reprs = [repr(item) for item in modules]

    count = 1
    block = reprs[0]

    for r in reprs[1:]:
        if r == block:
            count += 1

    return count, block


def process_layer(layer: nn.Module):
    ret = {}

    for child in layer.children():
        lowered_name = mname(child).lower()
        sub = filter_dropout_module(child)
        size = len(sub)

        if size > 0:
            if "mlp" in lowered_name:
                ret["mlp"] = mlp(sub)
            elif "attention" in lowered_name:
                ret["attention"] = attention(sub, size)
            else:
                pass
        else:
            if "input_layernorm" in ret:
                ret["post_attention_layernorm"] = convert(child)
            else:
                ret["input_layernorm"] = convert(child)

    return ret
