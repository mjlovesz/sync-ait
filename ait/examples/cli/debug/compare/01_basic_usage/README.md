# Basic Usage


## 介绍
compare精度对比功能可以通过ait命令行方式启动。


## 运行示例
**不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```
  - `-om, –om-model` 指定昇腾AI处理器的离线模型（.om）路径
  - `-gm, --golden-model` 指定模型文件（.pb或.onnx）路径
  - `-c，–-cann-path` (可选) 指定 `CANN` 包安装完后路径，默认为 `/usr/local/Ascend/ascend-toolkit/latest`
  - `-o, –-output` (可选) 输出文件路径，默认为当前路径


### 输出结果说明

```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape} 用来区分动态shape时不同的模型实际输入，静态shape时没有该层
├-- dump_data
│   ├-- npu                          # npu dump 数据目录
│   │   ├-- {timestamp}              # 模型所有npu dump的算子输出，dump为False情况下没有该目录
│   │   │   └-- 0                    # Device 设备 ID 号
│   │   │       └-- {om_model_name}  # 模型名称
│   │   │           └-- 1            # 模型 ID 号
│   │   │               ├-- 0        # 针对每个Task ID执行的次数维护一个序号，从0开始计数，该Task每dump一次数据，序号递增1
│   │   │               │   ├-- Add.8.5.1682067845380164
│   │   │               │   ├-- ...
│   │   │               │   └-- Transpose.4.1682148295048447
│   │   │               └-- 1
│   │   │                   ├-- Add.11.4.1682148323212422
│   │   │                   ├-- ...
│   │   │                   └-- Transpose.4.1682148327390978
│   │   ├-- {time_stamp}
│   │   │   ├-- input_0_0.bin
│   │   │   └-- input_0_0.npy
│   │   └-- {time_stamp}_summary.json
│   └-- {onnx or tf or caffe}        # 原模型 dump 数据存放路径，onnx / tf / caffe 分别对应 ONNX / Tensorflow / Caffe 模型
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                  # 随机输入数据，若指定了输入数据，则该文件不存在
├-- model
│   ├-- {om_model_name}.json
│   └-- new_{om_model_name}.onnx     # 把每个算子作为输出节点后新生成的 onnx 模型
├-- result_{timestamp}.csv           # 比对结果文件
└-- tmp                              # 如果 -m 模型为 Tensorflow pb 文件, tfdbg 相关的临时目录
```

### 比对结果说明
- **比对结果** 在文件 `result_{timestamp}.csv` 中，比对结果的含义与基础精度比对工具完全相同，其中每个字段的含义可参考 [CANN商用版/比对步骤（推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0039.html)
* 下面简要介绍说明结果信息：
  |                  OpType |  NPUDump | DataType | Address | GroundTruth | DataType | TensorIndex|Shape|Overflow|CosineSimilarity|...|MeanRelativeError|CompareFailReason|
  |------------------------:|---------:|---------:|--------:|------------:|---------:|-----------:|----:|-------:|---------------:|--:|----------------:|----------------:|
  |                      Sub|Sub_26Mul_28| float16 |    NaN |Sub_26,Mul_28|   float32|Sub_26Mul_28:output:0|[1,1,1,108]|NO|      1|...|         0.000364|                 |
如上所示的结果文件中主要关注以下几项:
 - [x] NPUDump:这个对应om模型中的算子,由于融合规则,可能会对应多个GPU/CPU算子
 - [x] DataType:一共有两个,一个是NPU侧的数据类型,一个是CPU/GPU侧的数据类型,二者有所不同,可能会有精度损失问题.
 - [x] GroundTruth:om算子所对应的onnx模型算子
 - [x] Overflow:数据是否出现上下溢.
 - [x] Error:CosineSimilarity, RelativeEuclideanDistance, ..., MeanRelativeError等为各类误差,主要需要看是否某一项超过阈值(即某项异常),若超过则需要重点关注.
 - [x] CompareFailReason:比对失败原因,误差可能会因为除零非法或者不对应等原因造成无法计算,变为NaN值,会列出详细原因.
### 比对结果分析
- **analyser 分析结果** 在调用结束后打印，在全部对比完成后，逐行分析数据，排除 nan 数据，输出各对比项中首个差距不在阈值范围内的算子。

  | 对比项目                  | 阈值   |
  | ------------------------- | ------ |
  | CosineSimilarity          | <0.99  |
  | RelativeEuclideanDistance | >0.05  |
  | KullbackLeiblerDivergence | >0.005 |
  | RootMeanSquareError       | >1.0   |
  | MeanRelativeError         | >1.0   |

  输出结果使用 markdown 表格显示
  ```sh
  2023-04-19 13:54:10(1005)-[INFO]Operators may lead to inaccuracy:

  |                   Monitor |  Value | Index | OpType | NPUDump | GroundTruth |
  |--------------------------:|-------:|------:|-------:|--------:|------------:|
  |          CosineSimilarity | 0.6722 |   214 |    Mul |   Mul_6 |       Mul_6 |
  | RelativeEuclideanDistance |      1 |   214 |    Mul |   Mul_6 |       Mul_6 |
  ```
