# Basic Usage

## 介绍

Convert模型转换工具依托AIE（Ascend Inference Engine）推理引擎，提供由ONNX模型转换至om模型的功能。

## 使用场景约束
1. 当前仅支持**Ascend310**以及**Ascend310P**平台的AIE转换；
2. 当前仅支持**FP16**精度下的模型转换
3. 当前已验证模型：Resnet50、DBNet、CRNN

## 运行示例

```shell
ait convert aie --golden-model resnet50.onnx --output-file resnet50.om --soc-version Ascend310P3
```

结果输出如下：
```shell
[INFO] Execute command:['./ait_convert', 'resnet50.onnx', 'resnet50.om', 'Ascend310P3']
[INFO] AIE model convert finished, the command: ['./ait_convert', 'resnet50.onnx', 'resnet50.om', 'Ascend310P3']
[INFO] convert model finished.
```
