# FAQ
## 1.运行时出现`Inner Error`类错误
出现Inner类错误多半是内核或者内存出现错误导致的。
* 内存类：
```
output size:90000000 from user add align:64 < op_size:xxxxxxxxxxx
```
这个错误是由于工具运行时默认`output size`为90000000而模型输出大小超出该值导致的。
解决方法：执行命令中加入`--output-size`并指定足够大小（如500000000），每个输出对应一个值。
**注意**：指定的大小不要过大，否则会导致内存不足无法分配。
* 内核类
```
TsdOpen failed, devId=0, tdt error=1[FUNC:startAicpuExecutor][FILE:runtime.cc][LINE:1673]
```
这个错误是AI Core使用失败导致的，解决方法是：
```
unset ASCEND_AICPU_PATH
```

## 2.使用locat功能时，出现`Object arrays cannot be loaded when allow_pickle=False`
- 该错误时由于模型执行时onnxruntime对onnx模型使用了算子融合导致某些中间节点没有真实dump数据导致的。
- **解决方法**是增加参数`--onnx-fusion-switch False`,关闭算子融合，使所有数据可用。

## 3.安装类问题
### 安装ait的compare组件时，出现skl2onnx组件安装失败的情况
- **解决方法**1：更换pip源，自行手动安装skl2onnx。
    命令：
    ```
    pip3 install skl2onnx==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --force-reinstall
    ```
    **解决方法**2: 直接安装wheel包[skl2onnx](https://pypi.tuna.tsinghua.edu.cn/packages/5e/59/0a47737c195da98d33f32073174b55ba4caca8b271fe85ec887463481f67/skl2onnx-1.14.1-py2.py3-none-any.whl)
