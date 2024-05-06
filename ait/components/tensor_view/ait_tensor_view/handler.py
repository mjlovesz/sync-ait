# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import splitext

import torch

from ait_tensor_view.atb import read_atb_data, write_atb_data
from ait_tensor_view.operation import SliceOperation, PermuteOperation
from ait_tensor_view.print_stat import print_stat


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
            if args.atb:
                write_atb_data(tensor, out_path)
            else:
                torch.save(tensor, out_path)
            print(f'Tensor saved successfully to {out_path}')
        except Exception as e:
            print(f"Error saving Tensor: {e}")
