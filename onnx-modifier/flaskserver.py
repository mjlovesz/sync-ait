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

import argparse

from flask import Flask, render_template, request, send_file
from server import register_interface


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='the port of the webserver. Defaults to 5000.')
    parser.add_argument('--debug', type=bool, default=False, help='enable or disable debug mode.')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    app = Flask(__name__, template_folder="static", static_url_path="/")
    
    @app.route('/')
    def index():
        return render_template('electron.html')

    register_interface(app, request, send_file)
    app.run(host='localhost', port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
