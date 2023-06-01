# ait debug compare功能使用指南

## 简介
compare一键式全流程精度比对（推理）功能将推理场景的精度比对做了自动化，适用于 TensorFlow 和 ONNX 模型，用户只需要输入原始模型，对应的离线模型和输入，输出整网比对的结果，离线模型为通过 ATC 工具转换的 om 模型，输入 bin 文件需要符合模型的输入要求（支持模型多输入）。

该功能使用约束场景说明，参考链接：[CANN商用版/约束说明（仅推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/atlasaccuracy_16_0035.html)


## 工具安装
- 工具安装请见 [ait一体化工具使用指南](../../../README.md)

## 使用方法
### 功能介绍
#### 使用入口
compare功能可以直接通过ait命令行形式启动精度对比。启动方式如下：

**不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```

#### 参数说明

  | 参数名                   | 描述                                       | 必选   |
  |-----------------------| ---------------------------------------- | ---- |
  | -gm，--golden-model    | 模型文件（.pb或.onnx)路径，目前只支持pb模型与onnx模型       | 是    |
  | -om，--om-model        | 昇腾AI处理器的离线模型（.om）                        | 是    |
  | -i，--input            | 模型的输入数据路径，默认根据模型的input随机生成，多个输入以逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin | 否    |
  | -c，--cann-path        | CANN包安装完后路径，默认为/usr/local/Ascend/ascend-toolkit/latest | 否    |
  | -o，--output           | 输出文件路径，默认为当前路径                           | 否    |
  | -s，--input-shape      | 模型输入的shape信息，默认为空，例如"input_name1:1,224,224,3;input_name2:3,300",节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称 | 否    |
  | -d，--device           | 指定运行设备 [0,255]，可选参数，默认0                  | 否    |
  | --output-nodes        | 用户指定的输出节点。多个节点用英文分号（;）隔开。例如:"node_name1:0;node_name2:1;node_name3:0" | 否    |
  | --output-size         | 指定模型的输出size，有几个输出，就设几个值。动态shape场景下，获取模型的输出size可能为0，用户需根据输入的shape预估一个较合适的值去申请内存。多个输出size用英文分号（,）隔开, 例如"10000,10000,10000" | 否    |
  | --advisor             | 在比对结束后，针对比对结果进行数据分析，给出专家建议 | 否    |
  | -dr，--dym-shape-range | 动态Shape的阈值范围。如果设置该参数，那么将根据参数中所有的Shape列表进行依次推理和精度比对。(仅支持onnx模型)<br/>配置格式为："input_name1:1,3,200\~224,224-230;input_name2:1,300"。<br/>其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。 <br/> | 否  |
  | --dump                | 是否dump所有算子的输出并进行精度对比。默认是True，即开启全部算子输出的比对。(仅支持onnx模型)<br/>使用方式：--dump False            | 否  |
  | --convert             | 支持om比对结果文件数据格式由bin文件转为npy文件，生成的npy文件目录为./dump_data/npu/{时间戳_bin2npy} 文件夹。使用方式：--convert True | 否    |
  | -cp, --custom-op      | 支持存在NPU自定义算子的模型进行精度比对，使用方式：--custom-op="op_nanme"，其中op_name代表onnx模型中，仅支持在NPU上运行的算子名称。[使用示例](../../../examples/cli/debug/compare/03_npu_custom_op) | 否    |

### 使用场景

请移步[compare使用示例](../../../examples/cli/debug/compare/)

