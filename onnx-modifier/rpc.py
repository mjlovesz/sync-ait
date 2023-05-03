
def rpc_run(is_electron, register_interface, args=None):
    if is_electron:
        pipe_run(register_interface)
    else:
        flask_run(register_interface, args)


def flask_run(register_interface, args):
    from flask import Flask, render_template, request, send_file
    app = Flask(__name__)
    register_interface(app, render_template, request, send_file)
    
    app.run(host=args.host, port=args.port, debug=args.debug)

def pipe_run(register_interface):
    import sys
    import json
    class PipeApp:
        def __init__(self, request) -> None:
            self.path_amp = dict()
            self.request = request

        def route(self, path, **kwargs):
            def regg(func): 
                self.path_amp[path] = func
                return func
            
            return regg
        
        def run(self):
            while True:
                msg_str = sys.stdin.readline()
                # {"msg":{"file":"D:\\amit\\onnx-modifier\\modified_onnx\\resnet34-v1-7.onnx"}, "path":"/open_model"}
                # {"msg":{"added_node_info":{},"node_states":{},"changed_initializer":{},"rebatch_info":{},"added_inputs":{},"input_size_info":{},"added_outputs":{},"node_renamed_io":{},"node_states":{},"node_changed_attr":{},"model_properties":{},"postprocess_args":{}}, "path":"/download"}
                # {"msg":{"path":"/exit"}
                msg_dict = json.loads(msg_str)
                path = msg_dict.get("path", "")
                if "/exit" == path:
                    break
                
                if path not in self.path_amp:
                    raise ValueError("not found path")

                msg_deal_func = self.path_amp[path]

                self.request.set_json(msg_dict.get("msg", dict()))
                msg_back, status = msg_deal_func()
                print(json.dumps(dict(msg=msg_back, status=status)), flush=True)

        def send_file(self, file_path):
            return dict(file=file_path), 200
        
    class Request:
        def __init__(self) -> None:
            self._json = dict()

        def set_json(self, json_msg):
            self._json = json_msg

        def get_json(self):
            return self._json
        
        @property
        def files(self):
            return self._json
    

    
    request = Request()
    app = PipeApp(request)

    register_interface(app, lambda x:x, request, app.send_file)
    app.run()
