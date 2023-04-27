import pandas as pd
from utils.log_util import logger


def read_excel(path=""):
    # 读取Excel文件
    excel = pd.ExcelFile(path)
    # 获取所有Sheet的名称
    sheets = excel.sheet_names

    api_dfs = dict()
    # 遍历每个Sheet
    for sheet in sheets:
        # 读取当前Sheet的数据为DataFrame对象
        df = excel.parse(sheet)
        # 获取列名列表
        column_names = df.columns.tolist()
        if sheet.startswith('Sheet'):  # 无效Sheet
            continue
        # 打印当前Sheet的名称和列名
        logger.debug(f'{sheet} {column_names}')
        api_dfs[sheet] = df
    return api_dfs


def write_excel(df_dict, path='output.xlsx'):
    # 创建一个 Excel 文件
    excel = pd.ExcelWriter(path)
    if df_dict['CUDA_APIS'].empty:
        del df_dict['CUDA_APIS']
    for key, df in df_dict.items():
        key = key.replace('/', '.')[-31:]  # 最大支持31个字符
        df.to_excel(excel, sheet_name=key, index=False)

    # 保存 Excel 文件
    excel.save()
    logger.info(f'Analysis result saved in {path}')


def append_excel(df, sheet_name, path='output.xlsx'):
    # 创建一个 Excel 文件
    excel = pd.ExcelWriter(path)
    df.to_excel(excel, sheet_name=sheet_name, index=False)

    # 保存 Excel 文件
    excel.save()
    logger.info(f'Excel saved in {path}')
