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

import sys
import logging
import os
import sys
import json
import tempfile
from urllib import parse

import onnx
from onnx_modifier import OnnxModifier


class RequestInfo:
    def __init__(self) -> None:
        self._json = dict()

    @property
    def files(self):
        return self._json

    def set_json(self, json_msg):
        self._json = json_msg

    def get_json(self):
        return self._json


class RpcServer:
    def __init__(self, tmp_dir) -> None:
        self.path_amp = dict()
        self.request = RequestInfo()
        self.msg_cache = ""
        self.msg_end_flag = 2
        self.max_msg_len_recv = 500 * 1024 * 1024
        self.tmp_dir = tmp_dir

    @staticmethod
    def send_message(msg, status, file, req_ind):
        return_str = json.dumps(dict(msg=msg, status=status, file=file, req_ind=req_ind))
        # 格式固定，\n\n>>\n msg \n\n
        sys.stdout.write("\n\n>>\n" + parse.quote(return_str) + "\n\n")
        sys.stdout.flush()
        return return_str

    def send_file(self, file, **kwargs):
        file_path = os.path.join(self.tmp_dir, "modified.onnx")
        file.seek(0)
        with open(file_path, "wb") as modified_file:
            modified_file.write(file.read())
        return dict(file=file_path), 200

    def route(self, path, **kwargs):
        def regg(func):
            self.path_amp[path] = func
            return func

        return regg

    def run(self):
        is_exit = False
        return_str = ""
        while not is_exit:
            msg_str = self._get_std_in()
            if not msg_str:
                continue
            
            logging.debug(os.getpid())
            logging.debug(msg_str.replace("\n", " "))

            try:
                msg_dict = json.loads(msg_str)
            except json.JSONDecodeError as ex:
                self.send_message(msg="exception:" + str(ex) + "\n" + msg_str, status=500, file=None, req_ind=1)
                continue

            req_ind = msg_dict.get("req_ind", 0)
            path = msg_dict.get("path", "")
            if "/exit" == path:
                self.send_message("byebye", 200, None, 0)
                is_exit = True
                break
            
            try:
                return_str = self._deal_msg(msg_dict, req_ind)
            except Exception as ex:
                logging.debug(os.getpid())
                logging.debug(str(ex))
                self.send_message(msg="exception:" + str(ex) + "\n" + msg_str, status=500, file=None, req_ind=req_ind)

            logging.debug(os.getpid())
            logging.debug(return_str)

    def _get_std_in(self):
        msg_recv = sys.stdin.readline()
        msg_recv = msg_recv.strip()
        if msg_recv == "":
            if self.msg_cache == "":
                return ""
            self.msg_end_flag -= 1
            if self.msg_end_flag > 0:
                return ""
        else:
            self.msg_end_flag = 2
            self.msg_cache += msg_recv
            if len(self.msg_cache) > self.max_msg_len_recv:
                raise ValueError("msg is too long to recv")
            return ""

        msg_str = self.msg_cache
        self.msg_cache = ""
        self.msg_end_flag = 2
        return parse.unquote(msg_str)

    def _deal_msg(self, msg_dict, req_ind):
        path = msg_dict.get("path", "")
        if path not in self.path_amp:
            raise ValueError("not found path")

        msg_deal_func = self.path_amp.get(path)
        if msg_deal_func is None:
            raise ValueError("invalid path")

        self.request.set_json(msg_dict.get("msg", dict()))
        msg_back, status = msg_deal_func()
        file = None
        if isinstance(msg_back, dict) and "file" in msg_back:
            file = msg_back["file"]
            del msg_back["file"]

        return self.send_message(msg=msg_back, status=status, file=file, req_ind=req_ind)


class ServerError(Exception):
    def __init__(self, msg, status) -> None:
        super().__init__()
        self._status = status
        self._msg = msg

    @property
    def status(self):
        return self._status

    @property
    def msg(self):
        return self._msg


class FileAutoClear:
    def __init__(self, file, clear_func=None) -> None:
        self._file = file
        self._clear_func = self.close if clear_func is None else clear_func
        self._is_auto_clear = True
    
    def __enter__(self):
        return self, self._file
    
    def __exit__(self, type, value, trace):
        if self._is_auto_clear:
            self._clear_func(self._file)
    
    def set_not_close(self):
        self._is_auto_clear = False

    @staticmethod
    def close(file):
        file_path = file.name
        if not file.closed:
            file.close()
        if os.path.islink(file_path):
            os.unlink(file_path)
        if os.path.exists(file_path):
            os.remove(file_path)


def modify_model(modifier, modify_info, save_file):
    modifier.modify(modify_info)
    if save_file is not None:
        save_file.seek(0)
        save_file.truncate()
        modifier.check_and_save_model(save_file)
        save_file.flush()


def onnxsim_model(modifier, modify_info, save_file):
    try:
        from onnxsim import simplify
    except ImportError as ex:
        raise ServerError("请安装 onnxsim", 599) from ex

    modify_model(modifier, modify_info, save_file)

    # convert model
    model_simp, _ = simplify(modifier.model_proto)

    modifier.reload(model_simp)
    if save_file is not None:
        save_file.seek(0)
        save_file.truncate()
        onnx.save(model_simp, save_file)


def call_auto_optimizer(modifier, modify_info, output_suffix, make_cmd):
    try:
        import auto_optimizer
    except ImportError as ex:
        raise ServerError("请安装 auto-optimizer", 599) from ex

    import subprocess
    
    with FileAutoClear(tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".onnx")) as (_, modified_file):
        opt_file_path = modified_file.name + output_suffix
        modify_model(modifier, modify_info, modified_file)
        modified_file.close()

        python_path = sys.executable
        cmd = make_cmd(py_path=python_path, in_path=modified_file.name, out_path=opt_file_path)

        out_res = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if out_res.returncode != 0:
            raise RuntimeError("auto_optimizer run error: " + str(out_res.returncode) + str(out_res) + " cmd: " + " ".join(cmd))

        return opt_file_path, out_res.stdout.decode()


def optimizer_model(modifier, modify_info, opt_tmp_file):
    def make_cmd(py_path, in_path, out_path):
        return [py_path, "-m", "auto_optimizer", "optimize", in_path, out_path]

    opt_file_path, msg = call_auto_optimizer(modifier, modify_info, ".opti.onnx", make_cmd)

    if os.path.exists(opt_file_path):
        with FileAutoClear(open(opt_file_path, "rb")) as (_, opt_file):
            modifier.reload(onnx.load_model(opt_file, onnx.ModelProto, load_external_data=False))

            if opt_tmp_file is not None:
                opt_tmp_file.seek(0)
                opt_tmp_file.truncate()
                opt_tmp_file.write(opt_file.read())

    return msg

def extract_model(modifier, modify_info, start_node_name, end_node_name, tmp_file):
    def make_cmd(py_path, in_path, out_path):
        return [py_path, "-m", "auto_optimizer", "extract", in_path, out_path,
                start_node_name, end_node_name]
    
    extract_file_path, msg = call_auto_optimizer(modifier, modify_info, ".extract.onnx", make_cmd)

    if os.path.exists(extract_file_path):
        tmp_file.seek(0)
        tmp_file.truncate()
        with FileAutoClear(open(extract_file_path, "rb")) as (_, extract_file):
            tmp_file.write(extract_file.read())

    return msg


def json_modify_model(modifier, modify_infos):
    for modify_info in modify_infos:
        path = modify_info.get("path")
        modify_info = modify_info.get("data_body")
        if path == "/download":
            modify_model(modifier, modify_info, None)
        elif path == '/onnxsim':
            onnxsim_model(modifier, modify_info, None)
        elif path == '/auto-optimizer':
            optimizer_model(modifier, modify_info, None)
        elif path == '/load-json':
            json_modify_model(modifier, modify_info)
        else:
            raise ServerError("unknown path", 500)


def register_interface(app, request, send_file, tmp_dir):
    @app.route('/open_model', methods=['POST'])
    def open_model():
        onnx_file = request.files.get("file")
        if isinstance(onnx_file, str):
            OnnxModifier.from_model_path(onnx_file)
        else:
            OnnxModifier.from_name_stream(onnx_file.filename, onnx_file.stream)

        return 'OK', 200

    @app.route('/download', methods=['POST'])
    def modify_and_download_model():
        modify_info = request.get_json()
        modifier = OnnxModifier.ONNX_MODIFIER
        modifier.reload()   # allow downloading for multiple times
        with FileAutoClear(tempfile.TemporaryFile(mode="w+b")) as (auto_close, tmp_file):
            modify_model(modifier, modify_info, tmp_file)

            auto_close.set_not_close()  # file will auto close in send_file 
            return send_file(tmp_file, download_name="modified.onnx")

    @app.route('/onnxsim', methods=['POST'])
    def modify_and_onnxsim_model():
        modify_info = request.get_json()

        modifier = OnnxModifier.ONNX_MODIFIER
        modifier.reload()   # allow downloading for multiple times
        with FileAutoClear(tempfile.TemporaryFile(mode="w+b")) as (auto_close, tmp_file):
            try:
                onnxsim_model(modifier, modify_info, tmp_file)
            except ServerError as error:
                return error.msg, error.status

            auto_close.set_not_close()  # file will auto close in send_file 
            return send_file(tmp_file, download_name="modified_simed.onnx")

    @app.route('/auto-optimizer', methods=['POST'])
    def modify_and_optimizer_model():
        modify_info = request.get_json()

        modifier = OnnxModifier.ONNX_MODIFIER
        modifier.reload()   # allow downloading for multiple times
        with FileAutoClear(tempfile.TemporaryFile(mode="w+b")) as (auto_close, opt_tmp_file):
            try:
                out_message = optimizer_model(modifier, modify_info, opt_tmp_file)
            except ServerError as error:
                return error.msg, error.status
            
            OnnxModifier.ONNX_MODIFIER.cache_message(out_message)

            if opt_tmp_file.tell() == 0:
                return "auto-optimizer 没有匹配到的知识库", 204

            auto_close.set_not_close()  # file will auto close in send_file 
            return send_file(opt_tmp_file, download_name="modified_opt.onnx")
    
    @app.route('/extract', methods=['POST'])
    def modify_and_extract_model():
        modify_info = request.get_json()

        modifier = OnnxModifier.ONNX_MODIFIER
        modifier.reload()   # allow downloading for multiple times
        with FileAutoClear(tempfile.TemporaryFile(mode="w+b")) as (auto_close, extract_tmp_file):
            try:
                extract_model(modifier, modify_info, 
                              modify_info.get("extract_start"), modify_info.get("extract_end"),
                              extract_tmp_file)
            except ServerError as error:
                return error.status, error.msg

            if extract_tmp_file.tell() == 0:
                return "未正常生成子网", 204

            auto_close.set_not_close()  # file will auto close in send_file 
            return send_file(extract_tmp_file, download_name="extracted.onnx")

    @app.route('/load-json', methods=['POST'])
    def load_json_and_modify_model():
        modify_infos = request.get_json()
        modifier = OnnxModifier.ONNX_MODIFIER
        modifier.reload()   # allow downloading for multiple times

        with FileAutoClear(tempfile.TemporaryFile(mode="w+b")) as (auto_close, tmp_file):
            tmp_modifier = OnnxModifier(modifier.model_name, modifier.model_proto)
            try:
                json_modify_model(tmp_modifier, modify_infos)
            except ServerError as error:
                return error.msg, error.status

            tmp_modifier.check_and_save_model(tmp_file)
            OnnxModifier.from_model_proto(modifier.model_name, tmp_modifier.model_proto)

            auto_close.set_not_close()  # file will auto close in send_file 
            return send_file(tmp_file, download_name="extracted.onnx")

    @app.route('/get_output_message', methods=['POST'])
    def get_out_message():
        return OnnxModifier.ONNX_MODIFIER.cache_message(), 200


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp_dir:
        server = RpcServer(tmp_dir)
        register_interface(server, server.request, server.send_file, tmp_dir)
        server.run()
