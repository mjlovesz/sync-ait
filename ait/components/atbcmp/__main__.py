import os
import re
import shutil
import argparse

from components.utils.parser import BaseCommand
from common.utils import str2bool, check_range, check_positive_integer, check_op, safe_string, check_exec_cmd
from library.initial import init_dump_task, clear_dump_task, check_ids_string, check_number_list

class CompareCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = None
    def add_arguments(self, parser):
        parser.add_argument(
            '--only-save-desc',
            '-sd',
            required=False,
            dest="save_desc",
            type=str2bool,
            default=False,
            help='0 When save tensor, 1 When only save tensor description instead of tensor')
        
        parser.add_argument(
            '--save-tensor-ids',
            '-ids',
            required=False,
            dest="ids",
            type=check_ids_string,
            default="",
            help='Save Tensor Ids')
        
        parser.add_argument(
            '--execute-range',
            '-er',
            required=False,
            dest="range",
            type=check_number_list,
            default="0,0",
            help='The range of saving tensor.Eg:0,10')
    
        parser.add_argument(
            '--save-operation-child',
            '-child',
            required=False,
            dest="child",
            type=str2bool,
            default=True,
            help='Dump all data of child operations if True, do nothing if False.Default True')
    
        parser.add_argument(
            '--save-time',
            '-time',
            required=False,
            dest="time",
            type=check_positive_integer,
            default=1,
            help='0 when only need dump data before execution, 1 when only need dump data after execution, 2 both.Default 1')
    
        parser.add_argument(
            '--operation-name',
            '-opname',
            required=False,
            dest="opname",
            type=safe_string,
            default=None,
            help='Operation names need to dump, default none')
        
        parser.add_argument(
            '--save-tiling',
            '-tiling',
            required=False,
            dest="tiling",
            type=str2bool,
            default=False,
            help='Dump all data of child operations if True, do nothing if False')
        
        parser.add_argument(
            '--exec',
            dest="exec",
            required=False,
            type=safe_string,
            default='',
            help='Exec command to run acltransformer model inference')
        
    def handle(self, args, **kwargs):
        if args.exec and check_exec_cmd(args.exec):
            init_dump_task(args)
            # 有的大模型推理任务启动后，输入对话时有提示符，使用subprocess拉起子进程无法显示提示符
            os.system(args.exec)
            clear_dump_task()
            return
        
def get_cmd_instance():
    help_info = "Ascend Transformer Boost Dump Tool."
    cmd_instance = CompareCommand("atbdump", help_info)
    return cmd_instance