#!/usr/bin/env python3.8
import os
import sys
import argparse
import logging
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

    global ONNX_MODIFIER
    ONNX_MODIFIER = OnnxModifier.from_name_stream(onnx_file.filename, onnx_file.stream)

    return 'OK', 200


@app.route('/download', methods=['POST'])
def modify_and_download_model():
    modify_info = request.get_json()
    
    ONNX_MODIFIER.reload()   # allow downloading for multiple times
    ONNX_MODIFIER.modify(modify_info)
    ONNX_MODIFIER.check_and_save_model()

    return 'OK', 200


@app.route('/onnxsmi', methods=['POST'])
def modify_and_onnxsmi_model():
    try:
        from onnxsim import simplify
        import onnx
        modify_info = request.get_json()
        
        ONNX_MODIFIER.reload()   # allow downloading for multiple times
        ONNX_MODIFIER.modify(modify_info)
        save_path = ONNX_MODIFIER.check_and_save_model(save_dir="modified_onnx")

        # convert model
        model_simp, check = simplify(ONNX_MODIFIER.model_proto)
        onnx.save(model_simp, save_path)
        return send_file(save_path)
    except ImportError as ex:
        return "请安装 onnxsim", 599


@app.route('/auto-optimizer', methods=['POST'])
def modify_and_optimizer_model():
    try:
        import auto_optimizer
        import onnx
        import subprocess
        modify_info = request.get_json()
        ONNX_MODIFIER.reload()   # allow downloading for multiple times
        ONNX_MODIFIER.modify(modify_info)
        save_path = ONNX_MODIFIER.check_and_save_model(save_dir="modified_onnx")

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
            return "OK", 204
    except ImportError as ex:
        return "请安装 auto-optimizer", 500


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
