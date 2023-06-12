# Extract Command


## 介绍
对模型进行子图切分。

```bash
ait debug surgeon extract [OPTIONS] INPUT_MODEL OUTPUT_MODEL START_NODE_NAME END_NODE_NAME
```

extract 可简写为ext

参数说明：

| 参数                    | 说明                                                                                  | 是否必选 |
|-----------------------|-------------------------------------------------------------------------------------|------|
| OPTIONS               | 额外参数。可取值：<br/>    -c/--is-check-subgraph：是否校验子图。启用这个选项时，会校验切分后的子图。                  | 否    |
| INPUT_MODEL           | 输入ONNX待优化模型，必须为.onnx文件。                                                             | 是    |
| OUTPUT_MODEL          | 切分后的子图ONNX模型名称，用户自定义，必须为.onnx文件。                                                    | 是    |
| START_NODE_NAME1,2... | 起始节点名称。可指定多个输入节点，节点之间使用","分隔。                                                       | 是    |
| END_NODE_NAME1,2...   | 结束节点名称。可指定多个输出节点，节点之间使用","分隔                                                        | 是    |
| SUBGRAPH_INPUT_SHAPE  | 额外参数。可指定截取子图之后的输入shape。多节点的输入shape指定按照以下格式，"input1:n1,c1,h1,w1;input2:n2,c2,h2,w2"。 | 否    |
| SUBGRAPH_INPUT_DTYPE  | 额外参数。可指定截取子图之后的输入dtype。多节点的输入dtype指定按照以下格式，"input1:dtype1;input2:dtype2"。           | 否    |
使用特别说明：为保证子图切分功能正常使用且不影响推理性能，请勿指定存在**父子关系**的输入或输出节点作为切分参数。

## 运行示例

```bash
ait debug surgeon extract origin_model.onnx sub_model.onnx "s_node1,s_node2" "e_node1,e_node2" --subgraph_input_shape="input1:1,3,224,224" --subgraph_input_dtype="input1:float16"
```

输出示例如下：

```bash
2023-04-27 14:32:33,378 - auto-optimizer-logger - INFO - Extract the model completed, model was saved in sub_model.onnx
```