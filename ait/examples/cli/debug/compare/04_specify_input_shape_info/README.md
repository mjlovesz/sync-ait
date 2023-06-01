# Specify Input Shape Info


## 介绍

指定模型输入的shape信息(动态场景必须输入)。

## 运行示例

1. 指定-s或--input-shape进行精度对比。
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -s "image:1,3,224,224"
  ```
如果模型为动态shape模型，则会以该-s输入的shape信息进行推理和精度对比。

2. 指定-dr或--dym-shape-range进行多个shape情况的精度对比。(优先级比-s,--input-shape更高)
  ```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -dr "image:1,3,224-256,224~226"
  ```
以上总共会进行6次精度对比流程，分别对输入为["image:1,3,224,224","image:1,3,224,225","image:1,3,224,226","image:1,3,256,224","image:1,3,256,225","image:1,3,256,226"]的情况进行了比较。