import os.path

import pandas as pd
from utils.excel import read_excel, write_excel
from utils.log_util import logger


class Advisor:
    def __init__(self, results, api_path):
        self.results = results
        self.api_df = self._api_map(api_path)
        self.api_key = os.path.basename(api_path)[:-9]

    @staticmethod
    def _api_map(api_path):
        df_dict = read_excel(api_path)
        # 将它们合并到一个DataFrame中
        apis = pd.concat([v for k, v in df_dict.items() if k.endswith('APIMap')], axis=0)
        cols = ['昇腾API', '说明', 'Opencv 4.5.4', '迁移预估人力（人/天）']
        drop_cols = [c for c in apis.columns if c not in cols]
        logger.debug(f'drop:{drop_cols}')
        apis = apis.drop(drop_cols, axis=1)
        apis['迁移预估人力（人/天）'].fillna(0.1, inplace=True)
        apis['Opencv 4.5.4'].fillna('', inplace=True)
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

    def recommend(self):
        for _, df in self.results.items():
            if df.empty:
                continue
            df[self.api_key] = ''
            df['Description'] = ''
            df['Workload'] = 0.0
            # 遍历每一行，并进行修改
            for index, row in df.iterrows():
                if not row['api']:
                    continue
                # 1. 使用Series.str.contains()做字符串检索
                # 2. 自定义字符串检索
                query = self._sort(row['api'], self.api_df, 'Opencv 4.5.4')

                if query:
                    best = query[0]
                    row[self.api_key] = best['昇腾API']
                    row['Description'] = best['说明']
                    row['Workload'] = best['迁移预估人力（人/天）']
                else:
                    row['Workload'] = 0.1
                df.iloc[index] = row

        return self.results

    def workload(self):
        wl = list()
        for file_name, df in self.results.items():
            if df.empty:
                continue
            workload = df['Workload'].sum()
            wl.append({'File': file_name, 'Workload': workload})
        wldf = pd.DataFrame(wl)
        total = wldf['Workload'].sum()
        wldf = pd.concat([wldf, pd.DataFrame({'File': ['Project'], 'Workload': [total]})], ignore_index=True)
        self.results['Workload'] = wldf
        return wldf

    def cuda_apis(self):
        cu_list = list()
        for file_name, df in self.results.items():
            if not df.empty and file_name != 'Workload':
                cu_list.append(df[df['cuda_en'] == True])
        cu_df = pd.concat(cu_list, ignore_index=True)
        cu_gp = cu_df.groupby('api').size()
        self.results['CUDA_APIS'] = pd.DataFrame({'api': cu_gp.index, 'count': cu_gp.values})
        return cu_gp

    def to_excel(self):
        write_excel(self.results)
