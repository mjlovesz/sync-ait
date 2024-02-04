import os
import sys
import re
import json
import queue
import threading
import csv
import time
import glob
import datetime
import pytz
import pandas as pd
import torch
import torch_npu

from llm.common.log import logger
from llm.opcheck.ut_manager import UtManager


class OpChecker:
    def __init__(self):
        '''
        cases_info结构：
            'op_id': string
            'op_name': string
            'op_param': dict
            'tensor_path': string
            'out_dtype: list
        '''
        self.csv_data = {}
        self.cases_info = {}
        self.completed_op_id_queue = queue.Queue()
        self.special_cases = ['KvCacheOperation', 'ReshapeAndCacheOperation', 'SelfAttentionOperation']
        self.tensor_path = ''
        self.op_path = ''
        self.output_dir = ''
        self.output_path = ''
        self.ids = ''
        self.check_ids_string = []
        self.opname = None
        self.check_patterns = []
        utc_time = datetime.datetime.now(tz=pytz.utc)
        self.timestamp = utc_time.astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")

    @staticmethod   
    def third_party_init():
        # LIB path设置
        lib_path = os.environ.get("AIT_OPCHECK_LIB_PATH")
        if lib_path and os.path.exists(lib_path):
            logger.info(lib_path)
            torch.classes.load_library(lib_path)
        else:
            raise RuntimeError("AIT_OPCHECK_LIB_PATH not exist")

        # 指定需要使用的npu设备
        device_id = os.environ.get("SET_NPU_DEVICE")
        if device_id is not None:
            torch.npu.set_device(torch.device(f"npu:{device_id}"))
        else:
            torch.npu.set_device(torch.device("npu:0"))

    def start_test(self, args):
        # 0.初始化
        OpChecker.third_party_init()
        self.args_init(args)
        ut_manager = UtManager(self.completed_op_id_queue)
        
        # 1.将csv文件中的算子信息添加到self.cases_info
        self.add_file_info_to_cases()
        result_info = 'excuted_information'

        for _, case_info in self.cases_info.items():
            # 2.将self.cases_info中的用例添加到ut_manager
            if_successed_add_case = ut_manager.add_case(case_info)
            if if_successed_add_case:
                case_info[result_info] = 'addition successed'
            else:
                case_info[result_info] = 'addition failed'

        # 3.执行测试用例并提供专家建议
        self.excute_cases(ut_manager)

        # 4.写入未添加成功的算子
        for v in self.cases_info.values():
            if v[result_info] == 'addition failed':
                v['res_detail'] = []
                self.write_op_result_to_csv(v)

        # 5.格式化文件
        data = pd.read_csv(self.output_path, dtype='str')
        data.to_excel(os.path.join(self.output_dir, f"opcheck_result_{self.timestamp}.xlsx"), index=False)

    def args_init(self, args):
        self.tensor_path = args.input
        self.op_path = args.csv_path
        self.output_dir = args.output
        self.output_path = os.path.join(self.output_dir, f"opcheck_result_{self.timestamp}.csv")
        self.ids = args.ids
        if self.ids != '':
            self.check_ids_string = [x.lower().strip() for x in self.ids.split(',')]
        self.opname = args.opname
        if self.opname is not None:
            self.check_patterns = [x.lower().strip() for x in self.opname.split(',')]

    def parse_in_tensor_path(self, ids):
        in_tensor_path = os.path.join(self.tensor_path, '_*/'.join(ids.split("_")) + '_*', "after")
        files = glob.glob(in_tensor_path)
        if not len(files) == 1:
            raise RuntimeError("{} could not find a dir!".format(in_tensor_path))
        return files[0]
    
    def parse_csv_files(self):
        try:
            df = pd.read_csv(self.op_path, sep='|')
        except Exception as e:
            logger_text = f"Cannot read csv file: {self.op_path}"
            logger.info(logger_text)
            raise e
        
        op_name_str = "OpName"
        if op_name_str in df.columns and "OutDType" in df.columns:
            try:
                df['Ids'] = df[op_name_str].apply(lambda x:x.split("_", 1)[1])
                df['RealOpName'] = df[op_name_str].apply(lambda x:x.split("_", 1)[0])
                df['InTensorPath'] = df['Ids'].apply(lambda x:self.parse_in_tensor_path(x))
                df['OutDTypeParse'] = df['OutDType'].apply(lambda x:x.split(";"))
            except Exception as e:
                logger_text = f"Cannot parse csv file: {self.op_path}"
                logger.info(logger_text)
                raise e
        else:
            raise RuntimeError("Cannot find enough info in csv file: {}".format(self.op_path))
        return df

    def check_id_range(self, op_id):
        if self.ids == '':
            return True
        else:
            for p in self.check_ids_string:
                ret = re.match("^" + p + "(_[1-9]+)*$", op_id)
                if ret:
                    return True
            return False
    
    def check_name(self, op_name):
        if self.opname is None:
            return True
        else: # 应该是LinearOps，SelfAttention
            for p in self.check_patterns:
                if p in op_name.lower():
                    return True        
            return False

    def if_exec_node(self, row):
        if self.ids == '' and self.opname is None:
            return True
            
        flag1 = self.check_id_range(row["Ids"])
        flag2 = self.check_name(row["RealOpName"])
        if flag1 and flag2:
            return True
        
        return False

    def add_case_to_cases_info(self, row):
        op_id = row['Ids']
        op_name = row['RealOpName']
        try:
            op_param = json.loads(row['OpParam'])
        except TypeError as e:
            op_param = {}

        tensor_path = row["InTensorPath"]
        out_dtype = row["OutDTypeParse"]

        case_info = {
            'op_id': op_id, 'op_name': op_name, 'op_param': op_param, 'tensor_path': tensor_path, 
            'out_dtype':out_dtype
        }

        if op_name == 'KvCacheOperation':
            case_info['inplace_idx'] = [2]
            self.cases_info[op_id] = case_info
        elif op_name == 'ReshapeAndCacheOperation':
            case_info['inplace_idx'] = [2, 3]
            self.cases_info[op_id] = case_info
        elif op_name == 'SelfAttentionOperation':
            self.cases_info[op_id] = case_info
        else:
            self.cases_info[op_id] = case_info 

    def add_file_info_to_cases(self):
        if os.path.exists(self.op_path):
            csv_data = self.parse_csv_files()

            for _, row in csv_data.iterrows():
                flag = self.if_exec_node(row)
                if flag:
                    self.add_case_to_cases_info(row)    
        else:
            raise RuntimeError(f"{op_path} not valid")
 
    def excute_cases(self, ut_manager):
        # 定义监控队列函数
        def watching_queue():
            cases_num = len([1 for v in self.cases_info.values() if v["excuted_information"] == 'addition successed'])
            cases_index = 0
            while cases_index < cases_num:
                time.sleep(0.1)
                if not self.completed_op_id_queue.empty():
                    completed_op_id = self.completed_op_id_queue.get()
                    case_info = self.cases_info.get(completed_op_id, '')
                    if case_info != '':
                        self.write_op_result_to_csv(case_info)
                    cases_index += 1
                    logger_text = f"===============excuted cases:{cases_index}, total cases:{cases_num}================"
                    logger.info(logger_text)

        watching_thread = threading.Thread(target=watching_queue)
        watching_thread.start()      
        ut_manager.excute_cases()
        watching_thread.join()
    
    def write_op_result_to_csv(self, op_result):
        with open(self.output_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if csv_file.tell() == 0:
                writer.writerow(['op_id', 'op_name', 'op_param', 'tensor_path', 'out_tensor_id', 
                                'precision_standard', 'precision_result(%)', 'excuted_information'])

            op_id = op_result['op_id']
            op_name = op_result['op_name']
            op_param = op_result['op_param']
            tensor_path = op_result['tensor_path']
            excuted_information = op_result['excuted_information']
            if len(op_result['res_detail']) > 0:
                for i, res_detail in enumerate(op_result['res_detail']):
                    precision_standard = res_detail['precision_standard']
                    rel_error_rate = res_detail['rel_error_rate']
                    writer.writerow([op_id, op_name, op_param, tensor_path, i, precision_standard, 
                                    rel_error_rate, excuted_information])
            else:
                default_str = 'NA'
                i = default_str
                precision_standard = default_str
                rel_error_rate = default_str
                writer.writerow([op_id, op_name, op_param, tensor_path, i, precision_standard, 
                                rel_error_rate, excuted_information])