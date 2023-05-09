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

import os
import sys
import argparse
import logging

import onnx
from onnx_modifier import OnnxModifier
from rpc import rpc_run


class ServerError(Exception):
    def __init__(self, msg, status) -> None:
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
        raise ServerError("请安装 onnxsim", 599)
        
    save_path = modify_model(modify_info)

    # convert model
    model_simp, check = simplify(OnnxModifier.ONNX_MODIFIER.model_proto)
    onnx.save(model_simp, save_path)
    return save_path


def optimizer_model(modify_info):
    try:
        import auto_optimizer
    except ImportError as ex:
        raise ServerError( "请安装 auto-optimizer", 599)
    
    import subprocess
    OnnxModifier.ONNX_MODIFIER.modify(modify_info)
    save_path = OnnxModifier.ONNX_MODIFIER.check_and_save_model(save_dir="modified_onnx")

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
    out_res = subprocess.call(cmd, shell=False)
    if  out_res != 0:
        raise RuntimeError("auto_optimizer run error: " + out_res + " cmd: " + "".join(cmd))
    return optimized_path

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
            optimized_path = optimizer_model(modify_info)
            if os.path.exists(optimized_path):
                OnnxModifier.from_model_path(optimized_path, model_name)
                save_path = optimized_path
        elif path == '/load-json':
            save_path = json_modify_model(modify_info)
        else:
            raise ServerError(500, "unknown path")
    
    return save_path
    

def register_interface(app, render_template, request, send_file):
    @app.route('/')
    def index():
        return render_template('index.html')


    @app.route('/open_model', methods=['POST'])
    def open_model():
        # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
        onnx_file = request.files['file']
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
        OnnxModifier.ONNX_MODIFIER.reload()
        
        try:
            optimized_path = optimizer_model(modify_info)
        except ServerError as error:
            return error.status, error.msg
        
        if not os.path.exists(optimized_path):
            raise ServerError( "auto-optimizer 没有匹配到的知识库", 204)
        if modify_info.get("return_modified_file"):
            return send_file(optimized_path)
        else:
            return 'OK', 200

    @app.route('/load-json', methods=['POST'])
    def load_json_and_modify__model():
        modify_infos = request.get_json()
        OnnxModifier.ONNX_MODIFIER.reload()

        try:
            save_path = json_modify_model(modify_infos)
        except ServerError as error:
            return error.status, error.msg
        
        return send_file(save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5000, help='the port of the webserver. Defaults to 5000.')
    parser.add_argument('--debug', type=bool, default=False, help='enable or disable debug mode.')
    parser.add_argument('--flask', action="store_true", help='enable or disable electron mode.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    rpc_run(True, register_interface, args)


if __name__ == '__main__':
    main()
