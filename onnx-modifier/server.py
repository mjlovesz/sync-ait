# Copyright 2023 Huawei Technologies Co., Ltd
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

import sys
import logging
import os
import sys
import json
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
    def __init__(self) -> None:
        self.path_amp = dict()
        self.request = RequestInfo()
        self.msg_cache = ""
        self.msg_end_flag = 2
        self.max_msg_len_recv = 500 * 1024 * 1024

    @staticmethod
    def send_message(msg, status, file):
        return_str = json.dumps(dict(msg=msg, status=status, file=file))
        # 格式固定，\n\n>>\n msg \n\n
        sys.stdout.write("\n\n>>\n" + parse.quote(return_str) + "\n\n")
        sys.stdout.flush()
        return return_str

    @staticmethod
    def send_file(file_path):
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
                return_str, is_exit = self._deal_msg(msg_str)
            except Exception as ex:
                logging.debug(os.getpid())
                logging.debug(str(ex))
                self.send_message(msg="exception:" + str(ex) +
                                  "\n" + msg_str, status=500, file=None)

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
            self.msg_cache += msg_recv
            if len(self.msg_cache) > self.max_msg_len_recv:
                raise ValueError("msg is too long to recv")
            return ""

        msg_str = self.msg_cache
        self.msg_cache = ""
        self.msg_end_flag = 2
        return parse.unquote(msg_str)

    def _deal_msg(self, msg_str):
        msg_dict = json.loads(msg_str)
        path = msg_dict.get("path", "")
        if "/exit" == path:
            return self.send_message("byebye", 200, None), True

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

        return self.send_message(msg=msg_back, status=status, file=file), False


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


def modify_model(modify_info):
    OnnxModifier.ONNX_MODIFIER.modify(modify_info)
    return OnnxModifier.ONNX_MODIFIER.check_and_save_model()


def onnxsim_model(modify_info):
    try:
        from onnxsim import simplify
    except ImportError as ex:
        raise ServerError("请安装 onnxsim", 599) from ex

    save_path = modify_model(modify_info)

    # convert model
    model_simp, check = simplify(OnnxModifier.ONNX_MODIFIER.model_proto)
    onnx.save(model_simp, save_path)
    return save_path


def optimizer_model(modify_info):
    try:
        import auto_optimizer
    except ImportError as ex:
        raise ServerError("请安装 auto-optimizer", 599) from ex

    import subprocess
    OnnxModifier.ONNX_MODIFIER.modify(modify_info)
    save_path = OnnxModifier.ONNX_MODIFIER.check_and_save_model(
        save_dir="modified_onnx")

    # convert model
    optimized_path = f"{save_path}.opti.onnx"
    python_path = sys.executable
    cmd = [
        python_path,
        "-m",
        "auto_optimizer",
        "optimize",
        save_path,
        optimized_path,
    ]
    out_res = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if out_res.returncode != 0:
        raise RuntimeError("auto_optimizer run error: " +
                           out_res + " cmd: " + "".join(cmd))
    return optimized_path, out_res.stdout.decode()


def json_modify_model(modify_infos):
    model_name = OnnxModifier.ONNX_MODIFIER.model_name
    for modify_info in modify_infos:
        path = modify_info.get("path")
        modify_info = modify_info.get("data_body")
        if path == "/download":
            save_path = modify_model(modify_info)
        elif path == '/onnxsim':
            save_path = onnxsim_model(modify_info)
            if os.path.exists(save_path):
                OnnxModifier.from_model_path(save_path, model_name)
        elif path == '/auto-optimizer':
            optimized_path, _ = optimizer_model(modify_info)
            if os.path.exists(optimized_path):
                OnnxModifier.from_model_path(optimized_path, model_name)
                save_path = optimized_path
        elif path == '/load-json':
            save_path = json_modify_model(modify_info)
        else:
            raise ServerError(500, "unknown path")

    return save_path


def register_interface(app, request, send_file):

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

        OnnxModifier.ONNX_MODIFIER.reload()   # allow downloading for multiple times
        save_path = modify_model(modify_info)
        if isinstance(save_path, tuple):
            return save_path
        if modify_info.get("return_modified_file"):
            return send_file(save_path)
        else:
            return 'OK', 200

    @app.route('/onnxsim', methods=['POST'])
    def modify_and_onnxsim_model():
        modify_info = request.get_json()

        OnnxModifier.ONNX_MODIFIER.reload()   # allow downloading for multiple times
        try:
            save_path = onnxsim_model(modify_info)
        except ServerError as error:
            return error.status, error.msg

        if modify_info.get("return_modified_file"):
            return send_file(save_path)
        else:
            return 'OK', 200

    @app.route('/auto-optimizer', methods=['POST'])
    def modify_and_optimizer_model():
        modify_info = request.get_json()

        OnnxModifier.ONNX_MODIFIER.reload()   # allow downloading for multiple times
        try:
            save_path, out_message = optimizer_model(modify_info)
        except ServerError as error:
            return error.status, error.msg
        
        OnnxModifier.ONNX_MODIFIER.cache_message(out_message)

        if not os.path.exists(save_path):
            return "auto-optimizer 没有匹配到的知识库", 204

        if modify_info.get("return_modified_file"):
            return send_file(save_path)
        else:
            return 'OK', 200

    @app.route('/load-json', methods=['POST'])
    def load_json_and_modify_model():
        modify_infos = request.get_json()
        OnnxModifier.ONNX_MODIFIER.reload()

        try:
            save_path = json_modify_model(modify_infos)
        except ServerError as error:
            return error.status, error.msg

        return send_file(save_path)

    @app.route('/get_output_message', methods=['POST'])
    def get_out_message():
        return OnnxModifier.ONNX_MODIFIER.cache_message(), 200


if __name__ == '__main__':
    server = RpcServer()
    register_interface(server, server.request, server.send_file)
    server.run()
