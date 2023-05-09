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

import logging
import os
import sys
import json

def rpc_run(is_flask, register_interface, args=None):
    if is_flask:
        flask_run(register_interface, args)
    else:
        pipe_run(register_interface)


def flask_run(register_interface, args):
    from flask import Flask, render_template, request, send_file
    app = Flask(__name__)
    register_interface(app, render_template, request, send_file)
    
    app.run(host=args.host, port=args.port, debug=args.debug)

def pipe_run(register_interface):

    class PipeApp:
        def __init__(self, request) -> None:
            self.path_amp = dict()
            self.request = request

        @staticmethod
        def send_file(file_path):
            return dict(file=file_path), 200

        def route(self, path, **kwargs):
            def regg(func): 
                self.path_amp[path] = func
                return func
            
            return regg
        
        def run(self):
            msg_cache = ""
            msg_end_flag = 2
            while True:
                try:
                    msg_recv = sys.stdin.readline()
                    # {"msg":{"file":"D:\\amit\\onnx-modifier\\modified_onnx\\resnet34-v1-7.onnx"}, "path":"/open_model"}
                    # {"msg":{"file":"C:\\需求\\amit\\amit\\modified_onnx\\res.onnx"}, "path":"/open_model"}
                    # {"msg":{"added_node_info":{},"node_states":{},"changed_initializer":{},"rebatch_info":{},
                    #           "added_inputs":{},"input_size_info":{},"added_outputs":{},"node_renamed_io":{},
                    #           "node_states":{},"node_changed_attr":{},"model_properties":{},"postprocess_args":{}}, 
                    #   "path":"/download"}
                    # {"path":"/exit"}
                    msg_recv = msg_recv.strip()
                    if msg_recv == "":
                        if msg_cache == "":
                            continue
                        msg_end_flag -= 1
                        if msg_end_flag > 0:
                            continue
                    else:
                        msg_cache += msg_recv
                        continue

                    msg_str = msg_cache
                    msg_cache = ""
                    msg_end_flag = 2

                    logging.debug(os.getpid())
                    logging.debug(msg_str)

                    msg_dict = json.loads(msg_str)
                    path = msg_dict.get("path", "")
                    if "/exit" == path:
                        print("byebye", flush=True)
                        break
                    
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

                    return_str = json.dumps(dict(msg=msg_back, status=status, file=file))
                    logging.debug(os.getpid())
                    logging.debug(return_str)
                    print(return_str, flush=True)
                except Exception as ex:
                    logging.debug(os.getpid())
                    logging.debug(str(ex))
                    print(json.dumps(dict(msg="exception:" + str(ex) + "\n" + msg_str, status=500, file=None)), flush=True)
        
    class Request:
        def __init__(self) -> None:
            self._json = dict()
        
        @property
        def files(self):
            return self._json

        def set_json(self, json_msg):
            self._json = json_msg

        def get_json(self):
            return self._json
    

    
    request = Request()
    app = PipeApp(request)

    register_interface(app, lambda x:x, request, app.send_file)
    app.run()
