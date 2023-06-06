# Convert 工具使用指南

## 介绍

Convert模型转换工具依托AIE（Ascend Inference Engine）推理引擎，提供由ONNX模型转换至om模型的功能。

## 工具安装

- 工具安装请见 [ait一体化工具使用指南](../../README.md)
- 如果使用AIE做模型转换，需要安装AIE并完成环境变量的配置:
  1. 安装AIE  
  ```bash
  ./Ascend-cann-aie-api_6.3.RC2_linux-x86_64.run --install
  ```
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

| 参数                  | 说明                                                                                                                                                                                     | 是否必选 |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------- |
| -gm, --golden_model | 标杆模型输入路径，支持onnx模型                                                                                                                                                                      | 是       |
| -o, --output        | 输出文件，需要有后缀 .om, 当前支持基于 AIE(Ascend Inference Engine) 的模型转换                                                                                                                              | 是       |
| -s, --soc_version   | 芯片类型，AIE模型转换当前只支持Ascend310P和Ascend310, 不指定则会通过[acl](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclpythondevg/aclpythondevg_01_0008.html)接口获取 | 否       |

命令示例如下：

```shell
ait convert --model resnet50.onnx --output resnet50.om --soc_version Ascend310 
```

## 使用案例
请移步[convert工具使用示例](../../examples/cli/convert/)