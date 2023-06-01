# Save Output Data


## 介绍

默认情况下，输出保存在当前目录下timestamp(例如./20230601115623)中。可使用-o或--output指定输出目录。

## 运行示例

```sh
  ait debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -i /home/HwHiAiUser/result/test/input_0.bin -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
```
该场景下，输出会保存在/home/HwHiAiUser/result/test/{timestamp}中

## 输出目录结构说明
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
│   └-- {onnx or tf}                         # -m 模型为 .onnx 时，onnx dump 数据目录，Tensorflow 模型为 tf
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # 随机输入数据，若指定了输入数据，则该文件不存在
├-- model
│   ├-- {om_model_name}.json
│   └-- new_{om_model_name}.onnx             # 把每个算子作为输出节点后新生成的 onnx 模型
├-- result_{timestamp}.csv                   # 比对结果文件
└-- tmp                                      # 如果 -m 模型为 Tensorflow pb 文件, tfdbg 相关的临时目录
```