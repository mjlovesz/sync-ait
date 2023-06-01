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


### 比对结果分析
- **比对结果** 在文件 `result_{timestamp}.csv` 中，比对结果的含义与基础精度比对工具完全相同，其中每个字段的含义可参考 [CANN商用版/比对步骤（推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0039.html)
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