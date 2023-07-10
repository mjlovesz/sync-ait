# ait convert功能使用指南

## 简介

convert模型转换工具依托AIE（Ascend Inference Engine）推理引擎，提供由ONNX模型转换至om模型的功能。

## 工具安装

- 工具安装请见 [ait一体化工具使用指南](../../README.md)
- 如果使用convert做模型转换，需要在安装convert前安装AIE并完成环境变量的配置:
  1. 安装AIE  
  ```bash
  ./Ascend-cann-aie-api_{version}_linux-{arch}.run --install
  ```
  {version}为版本号；
  {arch} 根据环境架构 (x86_64, aarch64) 获取对应的软件包
  
  2. 设置相关环境变量
  ```bash
  export AIE_DIR=/xxx/Ascend-cann-aie-api/
  ```

## 工具使用

一站式ait工具使用命令格式说明如下：

```shell
ait convert [OPTIONS]
```

OPTIONS参数说明如下：

| 参数                  | 说明                                                       | 是否必选 |
|---------------------|----------------------------------------------------------|------|
| -gm, --golden-model | 标杆模型输入路径，支持onnx模型                                        | 是    |
| -of, --output-file  | 输出文件，需要有后缀 .om, 当前支持基于 AIE(Ascend Inference Engine) 的模型转换 | 是    |
| -soc, --soc-version | 芯片类型，AIE模型转换当前只支持Ascend310P3和Ascend910B3                 | 是    |

命令示例如下：

```shell
ait convert --golden-model resnet50.onnx --output-file resnet50.om --soc-version Ascend310P3 
```

## 使用案例
请移步[convert工具使用示例](../../examples/cli/convert/)

## 使用限制
1. 目前convert组件仅支持使用onnxsim后的模型；
2. 目前convert组件支持以下4个模型：

| 参数                  | 说明                                                       | 是否必选 |
|---------------------|----------------------------------------------------------|------|
| -gm, --golden-model | 标杆模型输入路径，支持onnx模型                                        | 是    |
| -of, --output-file  | 输出文件，需要有后缀 .om, 当前支持基于 AIE(Ascend Inference Engine) 的模型转换 | 是    |
| -soc, --soc-version | 芯片类型，AIE模型转换当前只支持Ascend310P3和Ascend910B3                 | 是    |




