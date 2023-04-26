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
from flask import Flask, render_template, request, send_file
from onnx_modifier import OnnxModifier

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/open_model', methods=['POST'])
def open_model():
    # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    onnx_file = request.files['file']
    OnnxModifier.from_name_stream(onnx_file.filename, onnx_file.stream)

    return 'OK', 200


@app.route('/download', methods=['POST'])
def modify_and_download_model():
    modify_info = request.get_json()
    
    OnnxModifier.ONNX_MODIFIER.reload()   # allow downloading for multiple times
    OnnxModifier.ONNX_MODIFIER.modify(modify_info)
    OnnxModifier.ONNX_MODIFIER.check_and_save_model()

    return 'OK', 200


@app.route('/onnxsmi', methods=['POST'])
def modify_and_onnxsmi_model():
    try:
        from onnxsim import simplify
    except ImportError as ex:
        return "请安装 onnxsim", 599
    modify_info = request.get_json()
    
    OnnxModifier.ONNX_MODIFIER.reload()   # allow downloading for multiple times
    OnnxModifier.ONNX_MODIFIER.modify(modify_info)
    save_path = OnnxModifier.ONNX_MODIFIER.check_and_save_model(save_dir="modified_onnx")

    # convert model
    model_simp, check = simplify(OnnxModifier.ONNX_MODIFIER.model_proto)
    onnx.save(model_simp, save_path)
    return send_file(save_path)


@app.route('/auto-optimizer', methods=['POST'])
def modify_and_optimizer_model():
    try:
        import auto_optimizer
    except ImportError as ex:
        return "请安装 auto-optimizer", 599
    
    import subprocess
    modify_info = request.get_json()
    OnnxModifier.ONNX_MODIFIER.reload()   # allow downloading for multiple times
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
    if os.path.exists(optimized_path):
        return send_file(optimized_path)
    else:
        return "Nothing changed", 204


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5000, help='the port of the webserver. Defaults to 5000.')
    parser.add_argument('--debug', type=bool, default=False, help='enable or disable debug mode.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
