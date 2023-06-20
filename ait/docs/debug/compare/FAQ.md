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


















