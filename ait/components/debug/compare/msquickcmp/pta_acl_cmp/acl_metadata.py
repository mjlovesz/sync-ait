# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import re

INVALID_MD5_CHAR = re.compile("[^0-9a-f]")
MD5_STR_LEN = 32


def is_md5_str(input_str):
    return len(input_str) == MD5_STR_LEN and not INVALID_MD5_CHAR.search(input_str)


def init_encoder_decoder_token_id(base_path):
    # For ascend-transformer-acceleration dumped data, the first Decoder data is in 0, but it's actually token_id==1
    # thread_xxx
    # ├ 0
    # │ ├ Encoder  # token_id = 0
    # │ └ Decoder  # token_id = 1
    # ├ 1
    # │ └ Decoder  # token_id = 2
    # ├ 2
    # │ └ Decoder  # token_id = 3
    tokens = {}
    encoder_token_label, encoder_token_source, encoder_token_target = "Encoder", "1", "0"
    for token in os.listdir(base_path):
        cur_token = os.path.join(base_path, token)
        if not os.path.isdir(cur_token) or not str.isdigit(token):
            continue
        tokens[str(int(token) + 1)] = [os.path.join(cur_token, ii) for ii in os.listdir(cur_token)]

    if encoder_token_source in tokens:
        tokens[encoder_token_target] = [ii for ii in tokens[encoder_token_source] if encoder_token_label in ii]
        tokens[encoder_token_source] = [ii for ii in tokens[encoder_token_source] if encoder_token_label not in ii]
    return tokens


def extract_md5_info_from_token_dump_data(token_path, global_dict=None):
    cur_token = {} if global_dict is None else global_dict
    for cur_path, _, files in os.walk(token_path):
        if len(files) == 0 or not "after" in cur_path:
            continue
        if any(["outTensor" in file_name for file_name in files]):  # Op level, not kernel level
            continue

        cur_path = os.path.realpath(os.path.abspath(cur_path))
        for file_name in files:
            if not is_md5_str(file_name):
                continue
            cur_token.setdefault(file_name, []).append(cur_path)
    return cur_token


def init_acl_metadata_by_dump_data(base_path):
    tokens = init_encoder_decoder_token_id(base_path)
    metadata = {}
    for token, token_pathes in tokens.items():
        cur_token = {}
        for token_path in token_pathes:
            extract_md5_info_from_token_dump_data(token_path, global_dict=cur_token)
        metadata[token] = cur_token
    return metadata