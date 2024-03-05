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
import subprocess

from components.utils.parser import BaseCommand
from llm.dump.initial import init_dump_task, clear_dump_task
from llm.opcheck.opchecker import OpChecker
from llm.common.utils import str2bool, check_positive_integer, check_device_integer, safe_string, check_exec_cmd, \
    check_ids_string, check_number_list, check_output_path_legality, check_input_path_legality
from llm.common.log import set_log_level
from llm.common.log import logger


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
            help='0 when only need dump data before execution, '
                 '1 when only need dump data after execution, 2 both.Default 1')
    
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
            help='Exec command to run acltransformer model inference.'
                 'E.g: --exec \"bash run.sh patches/models/modeling_xxx.py\" ')
        
        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')

        parser.add_argument(
            '--save-tensor-part',
            '-stp',
            required=False,
            dest="save_tensor_part",
            type=check_positive_integer,
            default=2,
            help='0 when only need dump intensor, '
                 '1 when only need dump outtensor, 2 both.Default 2')

        parser.add_argument(
            '--type',
            dest="type",
            required=False,
            nargs='+',
            default=['tensor'],
            choices=['model', 'layer', 'op', 'kernel', 'tensor', 'cpu_profiling', 'onnx'],
            help='dump type.')

    def handle(self, args, **kwargs):
        if args.exec:
            logger.info(f"About to execute command : {args.exec}")
            logger.warning("Please ensure that your execution command is secure.")
            init_dump_task(args)
            # 有的大模型推理任务启动后，输入对话时有提示符，使用subprocess拉起子进程无法显示提示符
            cmds = args.exec.split()
            subprocess.run(cmds, shell=False)
            clear_dump_task(args)
            return


class CompareCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs):
        parser.add_argument(
            '--golden-path',
            '-gp',
            dest="golden_path",
            required=True,
            type=check_input_path_legality,
            help='Golden data path. It supports directory or file.')

        parser.add_argument(
            '--my-path',
            '-mp',
            dest="my_path",
            required=True,
            type=check_input_path_legality,
            help='Compared data path. It supports directory or file.')

        parser.add_argument(
            '--log-level',
            '-l',
            dest="log_level",
            required=False,
            default="info",
            type=str,
            help='Log level, default info.')
        
        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')

        parser.add_argument(
            '--op-mapping-file',
            '-mf',
            dest="mapping_file",
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Operation mapping file directory.E.g:--op-mapping-file /xx/xxxx/xx')

    def handle(self, args, **kwargs):
        from llm.compare.torchair_acc_cmp import get_torchair_ge_graph_path

        set_log_level(args.log_level)
        torchair_ge_graph_path = get_torchair_ge_graph_path(args.my_path)
        if torchair_ge_graph_path is not None:
            from llm.compare.torchair_acc_cmp import acc_compare

            acc_compare(args.golden_path, args.my_path, args.output, torchair_ge_graph_path)
        else:
            from llm.compare.atb_acc_cmp import acc_compare

            acc_compare(args.golden_path, args.my_path, args.output, args.mapping_file)


class OpcheckCommand(BaseCommand):
    def add_arguments(self, parser, **kwargs):
        parser.add_argument(
            '--input',
            '-i',
            dest="input",
            required=True,
            type=check_input_path_legality,
            help='input directory.E.g:--input OUTPUT_DIR/PID_TID/0/')

        parser.add_argument(
            '--csv-path',
            '-c',
            dest="csv_path",
            required=True,
            type=check_input_path_legality,
            help='csv file path.E.g:--csv-path OUTPUT_DIR/ait_dump/operation_io_tensors/PID/operation_tensors_0.csv')

        parser.add_argument(
            '--output',
            '-o',
            dest="output",
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Data output directory.E.g:--output /xx/xxxx/xx')
            
        parser.add_argument(
            '--operation-ids',
            '-ids',
            required=False,
            dest="ids",
            type=check_ids_string,
            default="",
            help='Save Tensor Ids.E.g:-ids 24_1,2_3_5')

        parser.add_argument(
            '--operation-name',
            '-opname',
            required=False,
            dest="opname",
            type=safe_string,
            default=None,
            help='Operation names need to dump.E.g:-opname self,linear')

        parser.add_argument(
            '--precision-metric',
            '-metric',
            dest="metric",
            required=False,
            nargs='+',
            default=[],
            choices=['abs', 'kl', 'cos_sim'],
            help=' Output more results of other precision metrics.E.g:-metric abs kl cos_sim')

        parser.add_argument(
            '--device-id',
            '-device',
            required=False,
            dest="device_id",
            type=check_device_integer,
            default=0,
            help='Spicifies the NPU device to bu used.E.g.:-device 1')

    def handle(self, args, **kwargs):
        op = OpChecker()
        logger.info(f"===================Opcheck start====================")
        op.start_test(args)
        logger.info(f"===================Opcheck end====================")


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
    compare_cmd_instance = CompareCommand("compare", "Accuracy compare tool for large language model",
                                          alias_name="cc")
    opcheck_cmd_instance = OpcheckCommand("opcheck", "Operation check tool for large language model", 
                                          alias_name='oo')
    return LlmCommand("llm", llm_help_info, [dump_cmd_instance, compare_cmd_instance, opcheck_cmd_instance])
