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

from components.utils.parser import BaseCommand
from llm.common.utils import str2bool, check_positive_integer, safe_string, check_exec_cmd, \
                              check_ids_string, check_number_list, check_output_path_legality
from llm.dump.initial import init_dump_task, clear_dump_task


class DumpCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--only-save-desc',
            '-sd',
            required=False,
            dest="save_desc",
            action='store_true',
            default=False,
            help='0 When save tensor, 1 When only save tensor description instead of tensor')
        
        parser.add_argument(
            '--save-operation-ids',
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
            action='store_true',
            default=False,
            help='Dump all data of child operations if True, do nothing if False')
        
        parser.add_argument(
            '--exec',
            dest="exec",
            required=True,
            type=safe_string,
            default='',
            help='Exec command to run acltransformer model inference.E.g: --exec \"bash run.sh patches/models/modeling_xxx.py\" ')
        
        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            type=check_output_path_legality,
            help='Data output directory.E.g:--output /xx/xxxx/xx')

    def handle(self, args, **kwargs):
        if args.exec and check_exec_cmd(args.exec):
            init_dump_task(args)
            # 有的大模型推理任务启动后，输入对话时有提示符，使用subprocess拉起子进程无法显示提示符
            os.system(args.exec)
            clear_dump_task()
            return


class CompareCommand(BaseCommand):
    def add_arguments(self, parser):
        pass

    def handle(self, args, **kwargs):
        pass


class LlmCommand(BaseCommand):
    def __init__(self, name="", help_info="", children=None, has_handle=False, **kwargs):
        super().__init__(name, help_info, children, has_handle, **kwargs)

    def add_arguments(self, parser, **kwargs):
        return super().add_arguments(parser, **kwargs)

    def handle(self, args, **kwargs):
        return super().handle(args, **kwargs)

    
def get_cmd_instance():
    llm_help_info = "Large Language Model(llm) Debugger Tools."
    dump_cmd_instance = DumpCommand("dump", "Dump tool for ascend transformer boost", alias_name="dd")
    compare_cmd_instance = CompareCommand("compare", "Compare tool for large language model",
                                          alias_name="cc")
    return LlmCommand("llm", llm_help_info, [dump_cmd_instance, compare_cmd_instance])