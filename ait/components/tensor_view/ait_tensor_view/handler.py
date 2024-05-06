import os.path

import torch
from os.path import splitext

from read_atb_data import read_atb_data
from operation import SliceOperation, PermuteOperation
from print_stat import print_stat


def replace(in_path: str, out_path: str) -> str:
    in_ext = splitext(in_path)[1]
    out_ext = splitext(out_path)[1]

    if in_ext and not out_ext:
        out_path += in_ext

    return out_path


def handle_tensor_view(args):
    tensor = read_atb_data(args.bin)

    in_ext = splitext(args.bin)[1]

    if args.operations:
        for op in args.operations:
            tensor = op.process(tensor)

    print_stat(tensor)

    if args.print:
        print(tensor)

    if args.output:
        out_path = args.output
        out_ext = splitext(out_path)[1]

        try:
            if not out_ext and in_ext:
                out_path += in_ext
            torch.save(tensor, out_path)
            print(f'Tensor saved successfully to {out_path}')
        except Exception as e:
            print(f"Error saving Tensor: {e}")
