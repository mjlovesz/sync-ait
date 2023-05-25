# Aipp Model Compare


## 介绍

提供模型转换开启aipp参数的om模型与标杆模型进行精度比对的功能。

## 运行示例

### 准备工作
先使用[atc工具](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/inferapplicationdev/atctool/atctool_0001.html)重新转换一个算子不融合的om模型：
```sh
atc --framework --model=./resnet18.onnx --output=resnet18_bs8 --input_format=NCHW \
--input_shape="image:8,3,224,224" --log=debug --soc_version=Ascend310P3 \
--insert_op_config=aipp.config --fusion_switch_file=fusionswitch.cfg
```
其中fusionswitch.cfg内容如下：
```
{
    "Switch":{
        "GraphFusion":{
            "ALL":"off"
        },
        "UBFusion":{
            "ALL":"off"
        }
    }
}
```

### 命令行操作
  ```sh
  ait debug compare -gm ./resnet18.onnx -om ./resnet18_bs8.om -s "image:8,3,224,224"
  ```
-om参数请输入上述生成的算子不融合的om模型;-s为onnx模型输入的shape信息;如果需要指定输入，
使用-i参数指定om模型的输入(npy或者bin文件)。