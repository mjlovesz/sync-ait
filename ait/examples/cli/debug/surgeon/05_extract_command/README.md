# Extract Command


## 介绍
对模型进行子图切分。

```bash
ait debug surgeon extract [OPTIONS] INPUT_MODEL OUTPUT_MODEL START_NODE_NAME END_NODE_NAME
```

extract 可简写为ext

参数说明：

| 参数              | 说明                                                                 | 是否必选 |
|-----------------|--------------------------------------------------------------------| -------- |
| OPTIONS         | 额外参数。可取值：<br/>    -c/--is-check-subgraph：是否校验子图。启用这个选项时，会校验切分后的子图。 | 否       |
| INPUT_MODEL     | 输入ONNX待优化模型，必须为.onnx文件。                                            | 是       |
| OUTPUT_MODEL    | 切分后的子图ONNX模型名称，用户自定义，必须为.onnx文件。                                   | 是       |
| START_NODE_NAME | 起始节点名称。                                                            | 是       |
| END_NODE_NAME   | 结束节点名称。                                                            | 是       |

## 运行示例

```bash
ait debug surgeon extract origin_model.onnx sub_model.onnx node1 node2
```

输出示例如下：

```bash
2023-04-27 14:32:33,378 - auto-optimizer-logger - INFO - Extract the model completed, model was saved in sub_model.onnx
```