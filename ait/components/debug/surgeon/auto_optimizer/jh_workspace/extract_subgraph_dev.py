from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph

g = OnnxGraph.parse("mobilenetv2-10.onnx")
input_shape = "input:1,3, 224, 224"
input_dtype = "input: float16"

parsed_info1 = g._parse_input_info(input_shape)
parsed_info2 = g._parse_input_info(input_dtype)
print(parsed_info1)
print(parsed_info2)

g.extract_subgraph(
    ['Conv_0'], ['Conv_4'],
    'tested.onnx',
    input_shape="input:1,3, 224, 224",
    input_dtype="input: float16"
)