# 工具使用指南

## 介绍

模型支持度分析工具提供算子支持情况分析、算子定义是否符合约束条件和算子输入是否为空。

### 软件架构



## 工具安装

- 工具安装请见 [ait一体化工具使用指南](../../README.md)


## 工具使用

一站式ait工具使用命令格式说明如下：

```shell
ait convert [OPTIONS]
```

OPTIONS参数说明如下：

| 参数                  | 说明                                                                                                                                                                                     | 是否必选 |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------- |
| -gm, --golden_model | 标杆模型输入路径，支持onnx模型                                                                                                                                                                      | 是       |
| -o, --output        | 输出文件，需要有后缀**.om**当前只支持基于**AIE(Ascend Inference Engine)**的模型转换                                                                                                                          | 是       |
| --framework         | 模型类型，和[atc](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000041.html)参数一致，0：caffe，3：tensorflow，5：onnx                          | 否       |
| --weight            | 权重文件，输入模型是caffe时，需要传入该文件                                                                                                                                                               | 否       |
| -s, --soc_version   | 芯片类型，AIE模型转换当前只支持Ascend310P和Ascend310, 不指定则会通过[acl](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclpythondevg/aclpythondevg_01_0008.html)接口获取 | 否       |

命令示例及输出如下：

```shell
ait convert -gm resnet50.onnx -o resnet50.om -s Ascend310 
```

```shell
2023-05-11 11:23:25,824 INFO : convert model to json, please wait...
2023-05-11 11:23:28,210 INFO : convert model to json finished.
2023-05-11 11:23:29,997 INFO : try to convert model to om, please wait...
2023-05-11 11:23:35,127 INFO : try to convert model to om finished.
2023-05-11 11:23:36,321 INFO : analysis result has bean writted in /tmp/result.csv
2023-05-11 11:23:36,321 INFO : analyze model finished.
```

