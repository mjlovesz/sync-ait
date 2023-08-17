# 大模型加速库在线推理精度比对使用指导
大模型加速库精度比对是以PyTorch Ascend(pta)侧的数据作为基准数据，比对加速库(acl)推理的数据与pta数据之间的差异，辅助开发者找出加速库侧的问题Operation。
## 1. 比对level
加速库的Operation分为3个粒度：Op、Layer、Model，在加速库开发过程中pta侧代码的替换也会分为这3个粒度。
## 1.1 Op的替换
若是Op粒度的替换，pta侧可以获取Operation的输入/输出数据，一个Operation内部会有多个kernel，pta侧无法获取到Op内部kernel的数据，因此需要加速库侧提供数据。<br>

如果比较Operation的输入/输出数据的精度，则称为***high-level***。如果比较Operation内部kernel的输入/输出数据的精度，则称为***low-level***。

## 1.2 Layer的替换

若是layer粒度的替换，pta侧可以获取整个Layer的输入/输出数据，一个Layer内部会有多个operation或kernel，pta侧无法获取到Layer内部Operation或者kernel的数据，因此需要加速库侧提供数据。

如果比较Layer的输入/输出数据的精度，则称为***high-level***。如果比较Layer内部kernel的输入/输出数据的精度，则称为***low-level***。

## 1.3 Model的替换

若是model粒度的替换，pta侧可以获取整个Model的输入/输出数据，一个Model内部会有多个Layer或Operation，pta侧无法获取Model内部Layer或者Operation的数据，因此需要加速库侧提供数据。

如果比较Model的输入/输出数据的精度，则称为***high-level***。如果比较Model内部Layer或者Operation的输入/输出数据的精度，则称为***low-level***。

## 2. 接口介绍

## 2.1 API介绍

### 2.1 set_label

set_label(data_src, data_id, data_val, tensor_path)

接口描述：用于在模型pta侧代码中打标签，记录待比对数据的来源、id、值以及路径。

返回值：无。

| 参数名      | 含义                   | 是否必填 | 使用说明                                                     |
| ----------- | ---------------------- | -------- | ------------------------------------------------------------ |
| data_src    | 数据来源               | 是       | 数据类型：str。可选值：acl、pta。acl表示加速库的数据，pta表示PyTorch Ascend的数据，即基准数据。 |
| data_id     | 数据的id。             | 是       | 数据类型：str，通过接口gen_id()生成，id一致的数据表示成对比较的数据。 |
| data_val    | 数据的值。             | 否。     | 数据类型: torch.Tensor。当data_src是pta时，这个值是必填的。当data_src是acl时，若是high-level比对，需要必填。若是low-level比对，不需要填。 |
| tensor_path | 加速库侧数据dump的路径 | 否       | 数据类型：str。当data_src是acl时，进行low-level比对时，需要提供加速库侧dump的operation或kernel的数据路径。 |

### 2.2 gen_id

gen_id()

接口描述：根据时间戳生成数据的id。

返回值：data_id，str类型。

### 2.3 set_task_id

set_task_id()

接口描述：用于设置加速库侧dump的数据目录，建议在每轮对话开始前调用下，可以进行多轮对话的精度比对。

返回值：无。

## 2.2. 命令行介绍

使用格式：

```shell
ait debug compare aclcmp xx_args
```

可选参数如下：

| 参数名 | 含义                                                         |
| ------ | ------------------------------------------------------------ |
| --exec | 执行命令，用于拉起大模型推理脚本。建议使用bash xx.sh args或者python3 xx.py的方式拉起。 |

# 3. 使用示例

以chatglm-6b为例，介绍下如何使用加速库精度比对工具。

 