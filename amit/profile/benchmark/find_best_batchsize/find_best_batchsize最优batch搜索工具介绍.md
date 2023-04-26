# find_best_batchsize推理介绍

## 介绍
本文介绍AisBench推理工具中find_best_batchsize专项功能

输入：原始模型文件，支持onnx、pb、prototxt格式(仅在采用atc工具的模式下支持三种模型，采用aoe工具的模式只支持onnx)

输出：最优的吞吐率，最优的batchsize序号

## 使用环境与依赖
已安装开发运行环境的昇腾AI推理设备。

## 构建与安装
本专项功能包含在推理工具中，构建和安装遵从推理工具的构建和安装。详细过程请参见推理工具的README.md

## 运行说明
在安装好推理whl包后，即可按照如下流程进行搜索命令执行
1. 确定requirement中依赖是否执行，如果没有安装，则执行如下命令进行安装
    ```
    root@root:/home/aclruntime-aarch64# pip3 install -r ./requirements.txt
    ```

2. 确定是否设置了CANN包的环境变量，如果没有设置，请执行如下命令进行设置，注意CANN包路径如果安装在其他目录,需手动替换
    ```
    root@root:/home/aclruntime-aarch64# source  /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

3. 下载[amit](https://gitee.com/ascend/amit)工程代码到本地，进入目录--amit\profile\benchmark\find_best_batchsize，运行find_best_batchsize.sh 执行最优batch搜索命令操作

## 使用方法

 ### 运行指令
 #### 1 使用atc生成om（--aoe_mode 0）
 onnx模型
```
bash  ./find_best_batchsize.sh --model_path /home/model/resnet50/resnet50.onnx --input_shape_str actual_input_1:batchsize,3,224,224 --soc_version Ascend310 --max_batch_num 10 --aoe_mode 0
```
pb模型
```
bash  ./find_best_batch.sh --model_path /home/lcm/tool/atc_bert_base_squad/save/model/BERT_Base_SQuAD1_1_BatchSize_None.pb --input_shape_str "input_ids:batchsize,384;input_mask:batchsize,384;segment_ids:batchsize,384" --soc_version "Ascend310" --max_batch_num 4 --aoe_mode 0
```
prototxt模型
```
bash  ./find_best_batchsize.sh --model_path /home/lhb/model/resnet50.prototxt --weight_path /home/lhb/model/resnet50.caffemodel --input_shape_str data:batchsize,3,224,224 --soc_version Ascend310 --max_batch_num 4 --aoe_mode 0
```
 #### 2 使用aoe生成om（--aoe_mode 1）
- 只支持onnx模型
 onnx模型
```
bash  ./find_best_batchsize.sh --model_path /home/model/resnet50/resnet50.onnx --input_shape_str actual_input_1:batchsize,3,224,224 --soc_version Ascend310 --max_batch_num 10 --aoe_mode 1 --job_type 1
```

### 运行参数说明

| 参数名   | 说明                            |
| -------- | ------------------------------- |
| --model_path  | 推理模型路径，支持onnx、pb、prototxt格式           |
| --weight_path  | 推理模型权因子文件路径。可选。只针对 caffe模型           |
| --max_batch_num | 最大搜索batch范围。值越大，搜索时间越长。默认值64      |
| --input_shape_str  | 推理模型输入节点 用于传入atc模型转换工具input_shape参数，格式为 name:shape;name1:shape1，同时需要将bath维度修改为 batchsize常量，以便用于工具进行遍历搜寻最佳batch。举例  输入节点信息为 actual_input_1:1,3,224,224  那么需要设置为 actual_input_1:batchsize,3,224,224        |
| --soc_version | 推理卡类型。支持昇腾310卡和710卡，可取值“Ascend310”、“Ascend310P”                |
| --python_command | 搜索支持的python版本。默认取值python3.7      |
| --loop_count   | 推理次数。可选参数。默认1000 |
| --device_id   | 指定运行设备 [0,255]，可选参数，默认0 |
| --aoe_mode |是否用aoe工具生成om模型，1为采用aoe，0为采用atc，默认1|
| --job_type|aoe工具的调优方式，可取{1，2}，数字对应的调优方式参考aoe工具的官方文档|
| --help| 工具使用帮助信息                  |

### 执行结果

以resnet50.onnx最优搜索结果为例（throughput的结果不代表实际结果，仅供演示）：
#### atc 模式下
```
best_batchsize:1 throughput:687.0000
best_batchsize:2 throughput:887.0000
best_batchsize:3 throughput:987.0000
best_batchsize:4 throughput:887.0000
best_batchsize:5 throughput:787.0000
calc and best batchsize:3 best throughtput:987.0000
```
#### aoe 模式下
```
best_batchsize:1 throughput:687.0000
best_batchsize:2 throughput:887.0000
best_batchsize:4 throughput:987.0000
best_batchsize:8 throughput:887.0000
best_batchsize:16 throughput:787.0000
calc and best batchsize:4 best throughtput:987.0000
```
