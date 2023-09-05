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

import pandas as pd


class SeqAdvisor:
    def __init__(self, result, idx_dict):
        self.result = result
        self.api_idx_dict = idx_dict

    def recommend(self):
        data_dict = dict()
        for seq_desc, lib_seqs in self.result.items():
            content = []
            entry_api = seq_desc.entry_api.full_name
            src_seq = '-->'.join([_.full_name for _ in seq_desc.api_seq])

            i = 0
            for lib_seq, rate in lib_seqs.items():
                for j, dst_seq in enumerate(lib_seq.dst_seqs):
                    rec_seq = '-->'.join([self.api_idx_dict[_] for _ in dst_seq])
                    if j == 0:
                        label = str(i + 1) + '.' + lib_seq.label
                        if i == 0:
                            item = [entry_api, src_seq, label, rec_seq, lib_seq.seq_desc[j], str(rate)]
                        else:
                            item = ['', '', label, rec_seq, lib_seq.seq_desc[j], str(rate)]
                    else:
                        item = ['', '', '', rec_seq, lib_seq.seq_desc[j], str(rate)]

                    content.append(item)
                i += 1

            if not lib_seqs:
                content = [[entry_api, src_seq, '', '', '', '']]

            loc = seq_desc.entry_api.location.file.name
            if data_dict.get(loc, None):
                data_dict[loc] += content
            else:
                data_dict[loc] = content

        df_dict = dict()
        for f, data in data_dict.items():
            df_dict[f] = pd.DataFrame(data,
                                      columns=['Entry API', 'Usr Call Seqs', 'Seq Labels', 'Recommended Sequences',
                                               'Functional Description', 'Recommendation Index'
                                               ],
                                      dtype=str)

        return df_dict
