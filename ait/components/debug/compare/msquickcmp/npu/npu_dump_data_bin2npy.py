import os
import sys

from common import utils

MSACCUCMP_FILE_PATH =  "toolkit/tools/operator_cmp/compare/msaccucmp.py"

def data_convert(npu_dump_data_path, npu_net_output_data_path, arguments):
    """
    Function Description:
        provide the interface for dump data conversion
    Exception Description:
        when invalid msaccucmp command throw exception
    """
    if _check_convert_bin2npy(arguments):
        common_path = os.path.commonprefix([npu_dump_data_path, npu_net_output_data_path])
        npu_dump_data_path_diff = os.path.relpath(npu_dump_data_path, common_path)
        time_stamp_file_path = npu_dump_data_path_diff.split(os.path.sep)[1]
        convert_dir_path = npu_dump_data_path.replace(time_stamp_file_path, time_stamp_file_path+'_bin2npy')
        convert_dir_path = os.path.normpath(convert_dir_path)
        convert_data_path = _check_data_convert_file(convert_dir_path)
        msaccucmp_command_file_path = os.path.join(arguments.cann_path, MSACCUCMP_FILE_PATH)
        python_version = sys.executable.split('/')[-1]
        bin2npy_cmd = [python_version, msaccucmp_command_file_path,"convert","-d",npu_dump_data_path,"-out",convert_data_path]
        utils.execute_command(bin2npy_cmd)
        utils.print_info_log("msaccucmp command line: %s "%" ".join(bin2npy_cmd))
        
def _check_data_convert_file(convert_dir_path):
    if not os.path.exists(convert_dir_path):
        os.makedirs(convert_dir_path)
        return convert_dir_path

def _check_convert_bin2npy(arguments):
    return arguments.bin2npy and arguments.dump