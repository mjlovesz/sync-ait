import json
import shlex
import os
import re
import math
import shutil
from datetime import datetime
import subprocess
import time
import locale

from utils.log_util import logger

TIMEOUT = 60 * 60 * 24


class IOUtil:
    """
    Define common utils
    """
    _ASM_ENV_CHECK_FLAG = 0
    _ASM_ENV_CHECK_RESULT = ""

    _WAIT_TIME = 5

    @staticmethod
    def check_required_cmd(command):
        """
        检测系统是否存在命令
        :param command:
        :return:
        """
        path = shutil.which(command)
        if path:
            return True
        return False

    @staticmethod
    def remove_dir(full_pathname):
        """扫描结束对解压目录进行删除操作"""
        shutil.rmtree(full_pathname, True)

    @staticmethod
    def remove_file(full_filename):
        os.remove(full_filename)

    @staticmethod
    def calc_workload(c_lines, asm_lines, c_lines_p, asm_lines_p):
        """calculate workload"""
        try:
            workload = math.ceil(10 * ((c_lines / c_lines_p) + (asm_lines / asm_lines_p))) / 10
        except ZeroDivisionError as error:
            logger.warning("Divisor cannot be zero. Except:%s.", error)
            workload = 0
        workload = '{:g}'.format(workload)
        return workload

    @staticmethod
    def remove_subdirectory(directories):
        """移除子目录"""
        if len(directories) == 0:
            return []
        flag = [0] * len(directories)
        for item1, item1_value in enumerate(directories[:-1]):
            if flag[item1]:
                continue
            for item2, item2_value in enumerate(directories[item1 + 1:],
                                                start=item1 + 1):
                if flag[item2]:
                    continue
                if os.path.commonprefix([item1_value, item2_value]) == \
                        item1_value:
                    flag[item2] = 1
                if os.path.commonprefix([item1_value, item2_value]) == \
                        item2_value:
                    flag[item1] = 1
        return [item_value for item, item_value in enumerate(directories) if not flag[item]]

    @staticmethod
    def escape_character(filename):
        """需要将包含特殊字符的文件名进行转义"""
        special_character = ('$', '?', '(', ')', '\\',
                             '|', '\'', '\"', '<', '>',
                             '!', '^', '*', '&', '[',
                             ']', '{', '}', ',', ':',
                             ';', '`', '=', ' ')
        return ''.join('_' if char in special_character else char
                       for char in filename)

    @staticmethod
    def transform_time_format(scan_time):
        """
        转换时间格式
        """
        scan_time = datetime.strptime(scan_time, '%Y%m%d%H%M%S')
        scan_time = datetime.strftime(scan_time, '%Y-%m-%d %H:%M:%S')
        return str(scan_time)

    @staticmethod
    def trans_format(input_format):
        """
        转换输入格式
        """
        if input_format:
            return ','.join(input_format) if isinstance(input_format, list) else input_format
        else:
            return 'None'

    @staticmethod
    def get_dir_size(path):
        """ 获取目录的大小 """
        cmd = "du --max-depth 0 -b %s" % path
        try:
            result = int(
                subprocess.check_output(
                    shlex.split(cmd)).decode().split()[0].strip()
            )
        except Exception as error:
            logger.warning("Failed to obtain the directory size. "
                           "Except:%s.", error)
            result = 0
        return result

    @staticmethod
    def get_file_or_dir_size(files_or_paths):
        """ 获取列表中所有文件的大小 """
        total = 0
        for file in files_or_paths:
            file = shlex.quote(file)
            if os.path.isfile(file):
                try:
                    total += os.path.getsize(file)
                except OSError as error:
                    logger.warning("Except:%s.", error)
            elif os.path.isdir(file):
                total += IOUtil.get_dir_size(file)
        return total

    @staticmethod
    def find_str_index(string, sub_str, find_cnt):
        list_str = string.split(sub_str, find_cnt)
        if len(list_str) <= find_cnt:
            return -1
        index = len(string) - len(list_str[-1]) - len(sub_str)
        return index

    @staticmethod
    def grep(text, keyword):
        return '\n'.join(
            filter(lambda line: line.find(keyword) != -1, text.split("\n")))

    @staticmethod
    def awk(text, index, sep=None):
        """
        文本切分和提取
        :param text: 原始文本
        :param index: 提取的索引（从0开始）
        :param sep: 分隔符，不传默认按照空白字符分隔
        :return:
        """
        try:
            if sep is not None:
                return text.split(sep)[index]
            return text.split()[index]
        except IndexError as error:
            logger.warning("The index value is incorrect. Except:%s.", error)
            return ""

    @staticmethod
    def read_file_contents(filepath):
        with open(filepath, errors='ignore') as file_desc:
            try:
                contents = file_desc.read()
                return contents
            except UnicodeDecodeError as error:
                logger.warning("Decoding failed. Except:%s.", error)
                return ''

    @staticmethod
    def get_file_path_in_rpm_or_deb(path):
        """
        获取解压后rpm或deb包中文件的路径（不包含包名）
        :param path: 依赖文件路径
        :return:
        """
        path_split = path.split('/')
        path_start_num = 1
        for path_item in path_split:
            if path_item.endswith(".rpm") or path_item.endswith(".deb"):
                break
            path_start_num += 1
        path = '/' + '/'.join(path_split[path_start_num:])
        return path

    @staticmethod
    def wait_last_task_complete(file_path):
        """
        等待上个任务完成，使用临时文件目录是否存在来判断，并置超时时间
        用于多任务环境下中止任务时删除文件和新建任务时拷贝文件带来的时序冲突
        :param file_path:
        :return:
        """
        count = IOUtil._WAIT_TIME
        while os.path.isdir(file_path) and count:
            count -= 1
            time.sleep(1)

    @staticmethod
    def write_to_log(path, build=False, error=False, success=False):
        """
        将BCGenerator生成BC文件过程中的信息写入porting.log
        :return:
        """
        build_file = os.path.join(path, "staticcodeanalyzer_build.log")
        error_file = os.path.join(path, "staticcodeanalyzer_error.log")
        if build:
            if not os.path.exists(build_file):
                return
            with open(build_file, "r") as build_obj:
                IOUtil.read_build_log(build_obj)
        if error:
            if not os.path.exists(error_file):
                return
            with open(error_file, "r") as error_obj:
                err = "".join(error_obj.readlines())
            logger.error("Failed to generate BC file. Except: %s", err)
        if success:
            for file in (build_file, error_file):
                if os.path.exists(file):
                    os.remove(file)

    @staticmethod
    def read_build_log(build_obj):
        while True:
            line = build_obj.readline()
            if line:
                logger.info(line.strip())
            else:
                break

    @staticmethod
    def get_barrier(json_file_path):
        """
        内存一致性静态检查后判断是否存在需要修改的文件
        :param json_file_path: json文件路径
        :return:
        """
        with open(json_file_path, 'r') as stream:
            json_data = json.load(stream)
        barriers = []
        for barrier in json_data["barriers"]:
            if isinstance(barrier["locs"], list):
                barrier["count"] = len(barrier["locs"])
                barriers.append(barrier)
        if barriers:
            barriers.sort(key=lambda x: x["file"])
        return barriers

    @staticmethod
    def fill_source_params(params):
        """
        填充源码扫描input必须的参数，用于非源码扫描模块调用构建文件解析
        :param params:
        :return:
        """
        params['compiler'] = 'gcc4.8.5'
        params['cgocompiler'] = 'gcc4.8.5'
        params['gfortran_version'] = 'gfortran7'
        params['interpreted'] = False
        params['target_os'] = 'centos7.6'
        params['target_kernel'] = '4.14.0'

    @staticmethod
    def get_upgrade_file(path, prefix, suffix=".tar.gz"):
        """
        获取迁移模板和依赖字典包名
        :param path: 路径/opt/portadv/portadmin或/opt/portadv/portadmin/migration
        :param prefix: 前缀，Porting-advisor-Dependency-dictionary-package或
            Porting-advisor-Migration-package
        :param suffix: 后缀，默认.tar.gz
        :return: 存在返回包名，不存在返回None
        """
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                if file_name.startswith(prefix) and file_name.endswith(suffix):
                    return file_path
        return ''

    @staticmethod
    def format_input_macros(custom_macros):
        """
        格式化用户输入的自定义宏
        :param custom_macros: 用户自定义宏
        :return: 格式化后的宏
        """
        marcos = re.split(r';+', custom_macros)
        filter_result = []
        for marco in marcos:
            marco = marco.strip()
            if not marco:
                continue
            filter_result.append(marco)
        result = ";".join(macro for macro in filter_result)
        return result

    @staticmethod
    def get_default_encoding():
        # 获取环境中当前的编码格式
        _, encoding = locale.getdefaultlocale()
        if not encoding:
            encoding = 'UTF-8'
        return encoding

    @staticmethod
    def get_uos_minor_version():
        """
        :return: 统信系统版本号
        """
        # 统信系统名称映射关系
        uos_name_map = {
            "1020": "UnionTech OS Server 20-1020",
            "1050": "UnionTech OS Server 20-1050"
        }
        if os.path.exists('/etc/os-version'):
            cmd = ['cat', '/etc/os-version']
            out = subprocess.check_output(cmd).decode()
            res = IOUtil.grep(out, "MinorVersion").split('=')[1]
        else:
            res = "1020"
        return uos_name_map.get(res)

    @staticmethod
    def check_path_is_empty(real_path):
        """检查文件夹是否为空文件夹"""
        if os.path.isdir(real_path):
            for _, _, files in os.walk(real_path):
                if files:
                    return False
            return True
        return not os.path.isfile(real_path)
