# ait benchmark API使用指南
## benchmark API简介
  目前benchmark提供了2种类型的python API可供使能基于昇腾硬件的离线模型(.om模型)推理。两种类型的API分别是：**interface API**和**aclruntime API**。
|接口类型|接口依赖|接口特点|
| ----- | ----- | ----- |
|interface API|python package: ais_bench, aclruntime|接口封装程度高，易用性好，快速上手|
|aclruntime API|python package: aclruntime|使用灵活，可以操作device侧数据|

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


## interface API
### 快速上手
InferSession 是单进程下interface API的主要类，它用于加载om模型和执行om模型的推理。
```python
from ais_bench.
```
## aclruntime API