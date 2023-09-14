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

import re
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
                match_seq = '-->'.join([self.api_idx_dict[_] for _ in lib_seq.src_seq])
                for j, dst_seq in enumerate(lib_seq.dst_seqs):
                    rec_seq = '-->'.join([self.api_idx_dict[_] for _ in dst_seq])
                    if j == 0:
                        label = str(i + 1) + '.' + lib_seq.label
                        if i == 0:
                            item = [entry_api, src_seq, label, rec_seq, lib_seq.seq_desc[j], str(rate), match_seq]
                        else:
                            item = ['', src_seq, label, rec_seq, lib_seq.seq_desc[j], str(rate), match_seq]
                    else:
                        item = ['', '', '', rec_seq, lib_seq.seq_desc[j], str(rate), '']

                    content.append(item)
                i += 1

            if not lib_seqs:
                content = [[entry_api, src_seq, '', '', '', '', '']]

            loc = seq_desc.entry_api.location.file.name
            if data_dict.get(loc, None):
                data_dict[loc] += content
            else:
                data_dict[loc] = content

        df_dict = dict()
        for f, data in data_dict.items():
            df = pd.DataFrame(data,
                              columns=['Entry API', 'Usr Call Seqs', 'Seq Labels', 'Recommended Sequences',
                                       'Functional Description', 'Recommendation Index', 'Tmp Match Sequences'
                                       ],
                              dtype=str)
            df_dict[f] = df
        return df_dict

    @property
    def format_fn(self):
        return self._postprocess

    @staticmethod
    def _postprocess(file_name, data, workbook):
        def _escape_string(word):
            escape_seq = ['<', '>', '.', '*', '(', ')', '[', ']']
            result = word
            for s in escape_seq:
                result = result.replace(s, '\\' + s)
            return result

        file_name = file_name.replace('/', '.')[-31:]  # 最大支持31个字符
        worksheet = workbook.add_worksheet(file_name)

        fmt = workbook.add_format({'color': 'red'})

        header = data.columns.values
        worksheet.write_row('A1', header[0:len(header) - 1])
        # 读取行数，后面对每行数据循环写入匹配到关键词的富字符串

        idx = data.columns.get_loc('Usr Call Seqs')
        for num in range(data.shape[0]):
            columns = [data.loc[num, _] for _ in header]

            src_call_seq = columns[idx]
            last_col = columns[-1]
            if not last_col:
                columns.pop(-1)
                for i, val in enumerate(columns):
                    worksheet.write(num + 1, i, val)
                continue

            match_str = ''  # 保存匹配到的关键词
            keywords = columns[-1].split('-->')
            for keyword in keywords:
                if src_call_seq.find(keyword) >= 0:
                    match_str = match_str + keyword + '|'

            # 虽然关键词已保存，但并未按其在文本中出现的位置顺序
            # 通过re.finditer按文本中出现的顺序匹配，所以进行二次匹配
            re_match = re.finditer(_escape_string(match_str.strip('|')), src_call_seq)

            re_str = ''
            for m in re_match:
                re_str += str(m.group()) + '|'

            keyword_split = re_str.strip('|').split('|')  # 这样就按文本中出现关键词的顺序列出了
            keyword_match = keyword_split
            match_words = []
            # 找出能模糊匹配到的字符长度短的关键词
            for j in range(len(keyword_split)):
                for k in range(len(keyword_match)):
                    if (keyword_split[j].find(keyword_match[k]) >= 0) and (
                            len(keyword_split[j]) > len(keyword_match[k])):
                        match_words.append(keyword_match[k])

            # 原关键词列表删除字符长度短的关键词
            for i in range(len(match_words)):
                if match_words[i] in keyword_split:
                    keyword_split.remove(match_words[i])

            keyword_set = keyword_split
            keyword_sep = ''
            for each in keyword_set:
                keyword_sep = keyword_sep + each + '|'
            keyword_sep = keyword_sep.strip('|')
            if not len(keyword_sep):
                columns.pop(-1)
                for i, val in enumerate(columns):
                    worksheet.write(num + 1, i, val)
                continue

            # 用所有关键词将整段话分割，再插入富字符串，然后捆绑颜色、关键词和后面的文本，需注意一一对应
            temp_list = re.split(_escape_string(keyword_sep), src_call_seq)
            params = []
            for i in range(len(temp_list)):
                if i != 0:
                    if temp_list[i] != '':
                        params.extend((fmt, keyword_set[i - 1], temp_list[i]))
                    else:
                        params.extend((fmt, keyword_set[i - 1]))
                else:
                    if temp_list[i] != '':
                        params.append(temp_list[i])

            columns.pop(-1)
            for i, val in enumerate(columns):
                if i == idx:
                    worksheet.write_rich_string(num + 1, i, *params)
                else:
                    worksheet.write(num + 1, i, val)
