# ait benchmark interface python API使用指南
## benchmark API简介
  benchmark提供的python API可供使能基于昇腾硬件的离线模型(.om模型)推理。<br>

使用ait benchmark 提供的api需要安装`ais_bench`和`aclruntime`包。安装方法有：
- 1、参考[一体化安装指导](https://gitee.com/ascend/ait/blob/master/ait/docs/install/README.md)安装ait benchmark工具
- 2、依据需求，单独安装ais_bench包和aclruntime包：
  ``` cmd
  # 安装aclruntime
  pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
  # 安装ais_bench
  pip3 install ./ais_bench-{version}-py3-none-any.whl
  # {version}表示软件版本号，{python_version}表示Python版本号，{arch}表示CPU架构。
  ```
## interface python API 快速上手
InferSession 是单进程下interface API的主要类，它用于加载om模型和执行om模型的推理，模型推理前需要初始化一个InferSession的实例。
```python
from ais_bench.infer.interface import InferSession

# InferSession的初始化表示在device id为0的npu芯片上加载模型model.om
session = InferSession(device_id=0, model_path="model.om")
```
建立好InferSession的实例session后，在npu芯片上进行模型推理所需的配置都已经完成，之后就可以直接调用session的成员函数接口进行模型推理，接口返回值就是推理结果。
```python
# feeds传入一组输入数据；mode选择模型类型，static表示输入节点shape固定的静态模型
# outputs 为ndarray格式的tensor
outputs = session.infer(feeds=inputs, mode="static")
```
推理结束，推理的性能数据也保存在session中，可以通过session的接口获取性能数据。
```python
# exec_time_list 按先后顺序保留了所有session在执行推理的时间。
exec_time = session.summary().exec_time_list[-1]
```
## interface python API 详细介绍
### InferSession
class <font color=#DD4466>**InferSession**</font>(<font color=#0088FF>device_id</font>: int, <font color=#0088FF>model_path</font>: str, <font color=#0088FF>acl_json_path</font>: str = None, <font color=#0088FF>debug</font>: bool = False, <font color=#0088FF>loop</font>: int = 1) <br>
$\qquad$ InferSession是**单进程**下用于om模型推理的类
#### 初始化参数
- **device_id**: npu芯片的id，在装了CANN驱动的服务器上使用`npu-smi info`查看可用的npu芯片的id。
- **model_path**: om模型的路径，支持绝对路径和相对路径。
- **acl_json_path**：acl json文件，用于配置profiling（采集推理过程详细的性能数据）和dump（采集模型每层算子的输入输出数据）。
- **debug**：显示更详细的debug级别的log信息的开关，True为打开开关。
- **loop**：一组输入数据重复推理的次数。

#### <font color=#DD4466>**get_inputs**</font>()
$\qquad$ **说明**: <br>
$\qquad\qquad$ 用于获取InferSession加载的模型的属于节点的信息，包括 <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**get_outputs**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**infer**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**infer_pipeline**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**infer_iteration**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**summary**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**reset_summaryinfo**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**free_resource**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**finalize**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

### MultiDeviceSession
class <font color=#DD4466>**MultiDeviceSession**</font>(<font color=#0088FF>model_path</font>: str, <font color=#0088FF>acl_json_path</font>: str = None, <font color=#0088FF>debug</font>: bool = False, <font color=#0088FF>loop</font>: int = 1) <br>
$\qquad$ MultiDeviceSession是**多进程**下用于om模型推理的类，初始化时不会在npu芯片(device)上加载模型，使用推理接口时才会在指定的几个devices的每个进程中新建一个InferSession。<br>
#### 初始化参数
- **model_path**
- **acl_json_path**
- **debug**
- **loop**

#### <font color=#DD4466>**infer**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**infer_pipeline**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**infer_iteration**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**summary**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**reset_summaryinfo**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

### MemorySummary
class <font color=#DD4466>**MemorySummary**</font>() <br>
$\qquad$ MemorySummary是。<br>

#### <font color=#DD4466>**get_h2d_time_list**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**get_d2h_time_list**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>

#### <font color=#DD4466>**reset**</font>()
$\qquad$ **说明**: <br>
$\qquad$ **参数**: <br>
$\qquad\qquad$ <font color=#0088FF>xx</font> <br>
$\qquad$ **返回值**: <br>
$\qquad\qquad$ <font color=#44AA00>xx</font> <br>