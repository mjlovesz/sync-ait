# 大模型精度评测工具API使用方法

## 导入依赖包
```python
from ais_bench.evaluate.interface import Evaluator
```
## Evaluator类原型
```python
class Evaluator(generate_func, dataset_name, dataset_path=None, shot=0, rank=0):
```


## 初始化参数
|参数名|说明|是否必选|
|----|----|----|
|**generate_func**|function object，封装了大模型推理能力的函数对象，输入自然语言，输出自然语言|是|
|**dataset_name**|str，评测数据集名称，目前支持ceval，mmlu，gsm8k|是|
|**dataset_path**|str，评测数据集的路径，支持绝对路径和相对路径。|否|
|**shot**|int，构造输入中的提示的数量，0<=shot<=5。|否|
|**rank**|int，多进程推理情况下进程的标号，只有在0号进程中对数据进行评测。|否|

## evaluate
**功能说明**

进行评测，可指定评测指标，目前支持accuracy和edit-distance。不指定评测指标时，选用默认的指标进行评测，默认指标和具体数据集相关。

**函数原型**
```python
evaluate(measurement=None)
```
**返回值**

None

## set_generate_func
**功能说明**

设置大模型推理函数

**函数原型**
```python
set_generate_func(generate_func)
```
**返回值**

None

## set_rank
**功能说明**

设置rank值

**函数原型**
```python
set_rank(rank)
```
**返回值**

None

## set_dataset
**功能说明**

设置数据集信息以及shot值。不指定数据集路径时，程序自动下载。

**函数原型**
```python
set_dataset(dataset_name, dataset_path=None, shot=0):
```
**返回值**

None