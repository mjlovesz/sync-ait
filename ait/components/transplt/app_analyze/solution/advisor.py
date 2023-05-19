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


class Advisor:

    def __init__(self, results):
        self.results = self._dedup_results(results)
        self.api_dfs = self._api_dfs(KitConfig.API_MAP)

    @staticmethod
    def _dedup_results(val_dict):
        """deduplicate scanning results caused by same include files in different source files"""
        rst_dict = {}
        df = pd.concat(list(val_dict.values()), ignore_index=True)
        df.drop_duplicates(subset=['API', 'Location'], keep='first', inplace=True)
        df['file'] = df['Location'].str.split(',', expand=True)[0]

        files = df['file'].unique()
        for f in files:
            tmp_df = df[df['file'].isin([f])]
            tmp_df = tmp_df.drop(columns='file')
            tmp_df.reset_index(drop=True, inplace=True)
            rst_dict[f] = tmp_df
        return rst_dict

    @staticmethod
    def _api_map(api_path):
        df_dict = read_excel(api_path)
        # 将它们合并到一个DataFrame中
        apis = pd.concat([v for k, v in df_dict.items() if k.endswith('APIMap')], axis=0)
        cols = ['昇腾API', '说明', 'NV_API', '迁移预估人力（人/天）']
        drop_cols = [c for c in apis.columns if c not in cols]
        logger.debug(f'drop:{drop_cols}')
        apis = apis.drop(drop_cols, axis=1)
        apis['迁移预估人力（人/天）'].fillna(0.1, inplace=True)
        apis['NV_API'].fillna('', inplace=True)
        return apis

    @staticmethod
    def _sort(api, df, col):
        scores = dict()
        rows = list()
        for _, row in df.iterrows():
            acc_apis = [s.strip() for s in row[col].split('/n')]
            if api in acc_apis:
                try:
                    scores[id(row)] = 1.0 / len(acc_apis)
                except ZeroDivisionError as ex:
                    raise ValueError("len(acc_apis) cannot be zero") from ex
                rows.append(row)
        rows.sort(key=lambda x: scores.get(id(x)))
        return rows

    @staticmethod
    def _workload_model(x):
        """工作量评估模型。"""
        # 或采用tanh模型：np.tanh(x / 15) * 15
        # 将定义域[0,30)缩放到[0,2)，对应的值域[0,0.5)
        try:
            y = (1 / (1 + np.exp(-x / 15)) - 0.5) * 2 * 15
        except ZeroDivisionError as ex:
            raise ValueError("workload_model encounters zero division error") from ex
        return np.ceil(y)

    def _api_dfs(self, api_map):
        api_dfs = dict()
        for k, v in api_map.items():
            lib_name = os.path.basename(v).split('_')[0]
            api_dfs[k] = [lib_name, self._api_map(v)]
        return api_dfs

    def recommend(self):
        for _, df in self.results.items():
            if df.empty:
                continue
            # 增加表格列
            df['AscendAPI'] = ''
            df['Description'] = ''
            df['Workload'] = 0.0
            df['AscendLib'] = ''
            df['Params(Ascend:Acc)'] = ''
            df['AscendAPI Link'] = ''
            df['AccAPI Link'] = ''
            # 遍历每一行，并进行修改
            for index, row in df.iterrows():
                if row['API'] in KitConfig.EXCEPT_API:
                    continue
                # 1. 使用Series.str.contains()做字符串检索
                # 2. 自定义字符串检索
                lib_name, api_df = self.api_dfs[row['AccLib']]
                query = self._sort(row['API'], api_df, 'NV_API')

                if query:
                    best = query[0]
                    row['AscendAPI'] = best['昇腾API']
                    row['Description'] = best['说明']
                    row['Workload'] = best['迁移预估人力（人/天）']
                    row['AscendLib'] = lib_name
                    row['Params(Ascend:Acc)'] = best.get('参数对应关系(昇腾:NV)', '')
                    row['AscendAPI Link'] = best.get('昇腾API链接', '')
                    row['AccAPI Link'] = best.get('NV_API链接', '')
                else:
                    row['Workload'] = 0.1
                df.iloc[index] = row

            drop_cols = list()
            for c in df.columns:
                k = c
                if c.startswith('Context'):
                    k = 'Context'
                if not KitConfig.OPTIONAL_REPORT_KEY.get(k, True):
                    drop_cols.append(c)
            logger.debug(f'drop:{drop_cols}')
            df.drop(drop_cols, axis=1, inplace=True)
        return self.results

    def workload(self):
        wl = list()
        for file_name, df in self.results.items():
            if df.empty:
                continue
            workload = df['Workload'].sum()
            wl.append({'File': file_name, 'Workload': workload, 'Rectified': self._workload_model(workload)})
        wldf = pd.DataFrame(wl)
        if wldf.empty:
            return wldf
        total = wldf['Workload'].sum()
        ttdf = pd.DataFrame({'File': ['Project'], 'Workload': [total], 'Rectified': self._workload_model(total)})
        wldf = pd.concat([wldf, ttdf], ignore_index=True)
        self.results['Workload'] = wldf
        return wldf

    def cuda_apis(self):
        cu_list = list()
        for file_name, df in self.results.items():
            if not df.empty and file_name != 'Workload':
                if 'CUDAEnable' not in df.columns:
                    continue
                cu_list.append(df[df['CUDAEnable'] == True])
        if not cu_list:
            return pd.DataFrame()
        cu_df = pd.concat(cu_list, ignore_index=True)
        cu_gp = cu_df.groupby('API').size()
        self.results['CUDA_APIs'] = pd.DataFrame({'API': cu_gp.index, 'Count': cu_gp.values})
        return cu_gp

    def to_excel(self):
        write_excel(self.results)
