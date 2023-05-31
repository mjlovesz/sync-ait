# 工具使用指南

## 介绍

模型支持度分析工具提供算子支持情况分析、算子定义是否符合约束条件和算子输入是否为空。

### 软件架构



## 工具安装

- 工具安装请见 [ait一体化工具使用指南](../../README.md)

如果使用AIE(Ascend Inference Engine)做模型转换，需要安装AIE，例如：./Ascend-cann-aie-api_6.3.RC2_linux-x86_64.run --install 对AIE进行安装。
安装完AIE后，需要设置AIE_DIR环境变量，指定安装的AIE版本路径，例如： export AIE_DIR=/xxx/Ascend-cann-aie-api/。


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

命令示例及输出如下：

```shell
ait convert --model resnet50.onnx --output resnet50.om --soc_version Ascend310 
```

