# Copyright (c) 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import numpy as np

from llm.common.log import logger

GE_GRAPH_FILE_PREFIX = "dynamo_original_graph_"

def set_msaccucmp_path_from_cann():
    cann_path = os.environ.get('ASCEND_TOOLKIT_HOME', "")
    msaccucmp_path = os.path.join(cann_path, "python", "site-packages", "operator_cmp", "compare")
    if not os.path.exists(msaccucmp_path):
        raise OSError("CANN toolkit in not installed or not set, try installing the latest CANN toolkit.")

    sys.path.append(msaccucmp_path)


def get_torchair_ge_dump_path(my_path):
    if not os.path.isdir(my_path):
        return None
    for ff in os.listdir(my_path):
        cur_file = os.path.join(my_path, ff)
        if os.path.isfile(cur_file) and ff.startswith(GE_GRAPH_FILE_PREFIX) and ff.endswith(".txt"):
            return cur_file
    return None


def is_torchair_dump_data(golden_data_path, my_path):
    return isinstance(golden_data_path, dict) and not my_path.endswith(".npy") and not my_path.endswith(".bin")


def parse_torchair_bin_dump_data(bin_dump_file):
    from dump_parser import DumpDataParser
    from dump_parse.big_dump_data import BigDumpDataParser
    from cmp_utils.constant.const_manager import ConstManager

    bin_dump_data = BigDumpDataParser(bin_dump_file).parse()  # Parser tool from CANN msaccucmp
    inputs = []
    for input_data in bin_dump_data.input:
        dtype = ConstManager.DATA_TYPE_TO_STR_DTYPE_MAP.get(input_data.data_type, "float32")
        inputs.append(np.frombuffer(input_data.data, dtype=dtype))

    outputs = []
    for output_data in bin_dump_data.output:
        dtype = ConstManager.DATA_TYPE_TO_STR_DTYPE_MAP.get(output_data.data_type, "float32")
        outputs.append(np.frombuffer(output_data.data, dtype=dtype))
    return inputs, outputs


def get_unique_key(cur_dict, cur_key, split_sign="#"):
    original_cur_key, cur_key_id = cur_key, 0
    while cur_key in cur_dict:
        cur_key_id += 1
        cur_key = original_cur_key + f"{split_sign}{cur_key_id}"
    return cur_key

def parse_pbtxt_to_dict(pbtxt_path):
    with open(pbtxt_path) as ff:
        contents = ff.read()

    result, cur_dict, superior_dicts, brackets_depth = [], {}, [], 0
    for cur_line in contents.split('\n'):
        cur_line = cur_line.strip()
        if len(cur_line) == 0:
            continue

        if " {" in cur_line:
            if brackets_depth == 0:
                cur_dict = {}
                superior_dicts = []
                result.append(cur_dict)
            cur_key = cur_line.split(" {")[0]
            cur_key = get_unique_key(cur_dict, cur_key)
            cur_dict[cur_key] = {}
            if len(superior_dicts) > brackets_depth:
                superior_dicts[brackets_depth] = cur_dict
            else:
                superior_dicts.append(cur_dict)
            cur_dict = cur_dict[cur_key]
            brackets_depth += 1
        elif ": " in cur_line:
            cur_key, cur_value = cur_line.split(": ")
            cur_key = get_unique_key(cur_dict, cur_key)
            cur_value = cur_value[1:-1] if cur_value.startswith('"') and cur_value.endswith('"') else cur_value
            cur_dict[cur_key] = cur_value
        elif "}" in cur_line:
            brackets_depth -= 1
            cur_dict = superior_dicts[brackets_depth]
    return result


def init_ge_dump_data_from_numpy_path(ge_dump_path):
    """
    For data like:
      npy_out/Mul.Mul_1.8.2.1706152333994087.input.0.npy
      npy_out/Mul.Mul_1.8.2.1706152333994087.input.1.npy
      npy_out/Mul.Mul_1.8.2.1706152333994087.output.0.npy

    Return dict:
      {'Mul_1': {
        'input': [
          'npy_out/Mul.Mul_1.8.2.1706152333994087.input.0.npy', 'npy_out/Mul.Mul_1.8.2.1706152333994087.input.1.npy'
        ],
        'output': ['npy_out/Mul.Mul_1.8.2.1706152333994087.output.0.npy']
      }}
    """
    ge_dump_data = {}
    for file_name in sorted(os.listdir(ge_dump_path)):
        split_name = file_name.split(".")
        is_input = split_name[-3] == "input"
        cur_op_map = ge_dump_data.get(split_name[1], {})
        cur_op_map.setdefault("input" if is_input else "output", []).append(os.path.join(ge_dump_path, file_name))
        ge_dump_data[split_name[1]] = cur_op_map
    return ge_dump_data


def gather_data_with_inference_id(data_path):
    gathered_files, cur_inference_id = {}, 0
    for cur_path, dirs, file_names in os.walk(data_path):
        if cur_path != data_path:
            cur_basename = os.path.basename(cur_path)
            cur_inference_id = int(cur_basename) if str.isdigit(cur_basename) else 0
        for file_name in file_names:
            gathered_files.setdefault(cur_inference_id, []).append(os.path.join(cur_path, file_name))
    return gathered_files

def init_ge_dump_data_from_bin_path(ge_dump_path):
    """
    For data like:
      1/Add.Add_2.44.6.1706596912161941,
      1/Cast.Cast_9.19.6.1706596911887829,
      1/ConcatV2D.ConcatV2.42.6.1706596912161117,

    Return dict:
      {1: {
            'Add_2': '1/Add.Add_2.44.6.1706596912161941',
            'Cast_9': '1/Cast.Cast_9.19.6.1706596911887829',
            'ConcatV2': '1/ConcatV2D.ConcatV2.42.6.1706596912161117',
      }}
    """
    gathered_files = gather_data_with_inference_id(ge_dump_path)

    dump_data_with_inference_id = {}
    for inference_id, file_list in gathered_files.items():
        cur_dump_data = {}
        for file_name in sorted(file_list):
            if file_name.endswith(".txt"):
                continue
            split_name = file_name.split(".")
            if len(split_name) < 5:
                logger.warning(f"invalid file name: {file_name}, should contain at least 4 '.'")
                continue

            cur_op_name = ".".join(split_name[1:-3])
            if cur_op_name in cur_dump_data:
                logger.warning(f"duplicated op name: {cur_op_name}")
                continue

            cur_dump_data[cur_op_name] = file_name
        dump_data_with_inference_id[inference_id] = cur_dump_data
    return dump_data_with_inference_id


def init_fx_dump_data_from_path(fx_dump_path):
    """
    For data like:
      1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy,
      1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy,
      1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy,

    Return dict:
      {1: {'mm-aten.mm.default': {
        'input': [
          '1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy',
          '1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy',
        ],
        'output': ['1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy']
      }}}
    """
    gathered_files = gather_data_with_inference_id(fx_dump_path)

    dump_data_with_inference_id = {}
    for inference_id, file_list in gathered_files.items():
        cur_dump_data = {}
        for file_path in sorted(file_list):
            file_name = os.path.basename(file_path)
            split_name = file_name.split(".")
            is_input = ".INPUT." in file_name
            cur_op_name = file_name.split('.INPUT.' if is_input else ".OUTPUT.")[0]
            cur_op_map = cur_dump_data.get(cur_op_name, {})
            cur_op_map.setdefault("input" if is_input else "output", []).append(file_path)
            cur_dump_data[cur_op_name] = cur_op_map
        dump_data_with_inference_id[inference_id] = cur_dump_data
    return dump_data_with_inference_id


def filter_valid_fx_desc_tensor_info(desc_key, desc_value):
    """Valid one like: 'attr': {'key': '_fx_tensor_name', 'value': {'s': 'add_1-aten.add.Tensor.OUTPUT.0'}}"""
    if not (desc_key == "attr" or desc_key.startswith("attr#")) or not isinstance(desc_value, dict):
        return False
    if desc_value.get("key", None) != "_fx_tensor_name" or not isinstance(desc_value.get("value", None), dict):
        return False
    if not isinstance(desc_value.get("value", {}).get("s", None), str):
        return False
    return True

def build_metadata_single_token(graph_map, ge_dump_data, fx_dump_data, token_id=0):
    metadata = {}
    data_id = token_id * len(graph_map)
    for cur_op in graph_map:
        op_info = cur_op.get("op", {})
        if op_info.get("name", None) not in ge_dump_data:
            continue

        cur_ge_data = ge_dump_data[op_info["name"]]
        for kk, vv in op_info.items():
            if not (kk == "output_desc" or kk.startswith("output_desc#")) or not isinstance(vv, dict):
                continue
            for out_kk, out_vv in vv.items():
                if not filter_valid_fx_desc_tensor_info(out_kk, out_vv):
                    continue
                fx_tensor_name = out_vv.get("value", {}).get("s", None)
                if fx_tensor_name.split(".")[-2] == "OUTPUT":
                    fx_tensor_name = ".".join(fx_tensor_name.split('.')[:-2])
                if not fx_tensor_name in fx_dump_data:
                    continue

                cur_fx_inputs = fx_dump_data.get(fx_tensor_name, {}).get("input", [])
                cur_fx_outputs = fx_dump_data.get(fx_tensor_name, {}).get("output", [])
                cur_map_item = [{"inputs": cur_fx_inputs, "outputs": cur_fx_outputs}, cur_ge_data]  # [golden, my]
                metadata[data_id] = {token_id: cur_map_item}
                data_id += 1
    return metadata


def build_metadata(graph_map, ge_dump_data, fx_dump_data):
    gathered_metadata = {}
    for token_id in ge_dump_data:
        if not token_id in fx_dump_data:
            logger.warning(f"GE token_id {token_id} not found in FX dump data")
            continue
        meta_data = build_metadata_single_token(graph_map, ge_dump_data[token_id], fx_dump_data[token_id], token_id)
        gathered_metadata.update(meta_data)
    return gathered_metadata
