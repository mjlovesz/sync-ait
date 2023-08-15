# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import os.path
import numpy as np

import pandas as pd
from app_analyze.utils.excel import read_excel, write_excel
from app_analyze.utils.log_util import logger
from app_analyze.common.kit_config import KitConfig


class SeqAdvisor:
    def __init__(self, result, idx_dict):
        self.result = result
        self.api_idx_dict = idx_dict

    def recommend(self):
        data_dict = dict()
        for seq_desc, lib_seqs in self.result.items():
            entry_api = seq_desc.entry_api.full_name
            src_seq = '-->'.join([_.full_name for _ in seq_desc.api_seq])
            dst_seq = ''
            func_desc = ''

            i = 1
            r_index = ''
            for lib_seq, rate in lib_seqs.items():
                acc_seq = list()
                for idx in lib_seq.dst_seq:
                    acc_seq.append(self.api_idx_dict[idx])

                dst_seq += (str(i) + '.' + '-->'.join(acc_seq) + '\n')
                func_desc += (str(i) + '.' + lib_seq.function + '\n')
                i += 1

                if not r_index:
                    r_index = str(rate)

            loc = seq_desc.entry_api.location.file.name
            if data_dict.get(loc, None):
                data_dict[loc].append([entry_api, func_desc, src_seq, dst_seq, r_index])
            else:
                data_dict[loc] = [[entry_api, func_desc, src_seq, dst_seq, r_index]]

        df_dict = dict()
        for f, data in data_dict.items():
            df_dict[f] = pd.DataFrame(data,
                                      columns=['Entry API', 'Function', 'Raw Sequence', 'Recommended Sequence',
                                               'Recommendation Index '],
                                      dtype=str)

        return df_dict
