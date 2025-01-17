torch-npu(gpu)模型推理数据dump

支持torch模型推理数据的dump，包括tensor，模型拓扑信息等，提供python API来使能数据dump。1.0版本仅支持torch.nn.Module类算子数据的dump，torch其他api数据dump将在后续版本中支持。

## 快速开始

修改原推理脚本，在模型初始化之后，插入如下代码：
```
# 推理模型已完成初始化 => model

from ait_llm import DumpConfig, register_hook
dump_config = DumpConfig(dump_path="./ait_dump")
register_hook(model, dump_config)  # model是要dump中间tensor的模型实例，在模型初始化后添加代码

# 继续原来的推理流程

```

## API说明

### DumpConfig

接口说明：dump数据配置类，可用于按需dump模型数据。

接口原型：DumpConfig(dump_path, token_range, module_list, tensor_part)

| 参数名      | 含义                   | 使用说明                                                     | 是否必填 | 版本 |
| ----------- | ---------------------- | ------------------------------------------------------------ | -------- | --|
| dump_path   | 设置dump的数据路径     | 数据类型：str，默认为当前目录。                              | 否       | 
| token_range | 需要dump的token列表    | 数据类型：list。默认为[0]，只dump第0个token的数据。          | 否       |
| module_list | 指定要hook的module类型 | 数据类型：list，默认为[]，即dump所有module的数据。           | 否       |
| tensor_part | 指定要dump哪部分数据   | 数据类型：int，默认为2。当tensor_part=0时，只dump输入数据；当tensor_part=1时，只dump输出数据； 当tensor_part=2时，dump输入和输出的数据。 | 否       |
| device_id   | 指定要dump的device id  | 数据类型：int，默认为None 表示不限制 device。如指定 device_id=1，将跳过其他 device 的 dump。 | 否       |
| dump_last_logits | 是否需要Dump 模型最后的输出logits | 数据类型： bool, 默认为False, 当开启后，仅 Dump 模型最后输出的 logits ,模型中间layer 不会再输出。 可参考 [《logits精度比对》](输出Token的logits精度比对-加速卡推理场景.md) | 否 |
| mode | 设置dump的模式 | 可以选择dump api 还是 module，默认是module，也可以传入数组['api', 'module']，表示两种都dump | 否 | 7.0.0b530 |
| dump_weight | 设置是否需要dump权重 | 数据类型:bool，默认是False，不dump。[dump落盘位置](#dump-落盘位置) | 否 | 7.0.0b530 |
| layer_name | 指定需要dump的layer名字 | 数据类型:str，可以通过该参数控制dump 的权重和tensor。支持 * 表示匹配0或多个随意字符，不支持其他的模式匹配 | 否 | 7.0.0b530 |

### register_hook

接口说明：给模型添加hook，用于dump数据

接口原型：register_hook(model, config, hook_type=”dump_data”)

| 参数名    | 含义           | 使用说明                                                | 是否必填 |
| --------- | -------------- | ------------------------------------------------------- | -------- |
| model     | 需要hook的模型 | 数据类型：torch.nn.Module，建议设置为最外层的torch模型  | 是       |
| config    | Hook配置       | 数据类型：DumpConfig                                    | 是       |
| hook_type | hook类型       | 数据类型：str，默认值为dump_data，当前仅支持dump_data。 | 否       |

### 更多使用示例


1. 仅dump 模型的第 0 个 block，假定第 0 个 block 的名字为：root.transformer.encoder.layers.0 。具体名字可以从model信息中获取
```
from ait_llm import DumpConfig, register_hook
dump_config = DumpConfig(dump_path="./ait_dump", layer_name='root.transformer.encoder.layers.0*')
register_hook(model, dump_config)
```

2. 仅dump 模型的权重
```
from ait_llm import DumpConfig, register_hook
dump_config = DumpConfig(dump_path="./ait_dump", dump_weight=True)
register_hook(model, dump_config)
```

3. 仅dump 模型的输出token的logits
```
from ait_llm import DumpConfig, register_hook
dump_config = DumpConfig(dump_path="./ait_dump", dump_last_logits=True)
register_hook(model, dump_config)
```

### dump 落盘位置

Dump默认落盘路径 `{DUMP_DIR}`在当前目录下，如果在DumpConfig中指定dump_path，落盘路径则为指定的 `{DUMP_PATH}`。

- tensor信息会生成在默认落盘路径的ait_dump目录下，具体路径是 `{DUMP_DIR}/ait_dump/torch_tensors/{device_id}_{PID}/{TID}`目录下
- model信息会生成在默认落盘路径的ait_dump目录下，具体路径是 `{DUMP_DIR}/ait_dump/torch_tensors/{device_id}_{PID}/model_tree.json`
- weights 权重信息回生成在默认落盘路径的ait_dump目录下，具体路径是`{DUMP_DIR}/ait_dump/torch_tensors/{device_id}_{PID}/weights` 目录下

> 注：`{device_id}`为设备号；`{PID}`为进程号；`{TID}`为 `token_id`

todo: 落盘数据是怎么样的，如何查看



## FAQ

1. WARNING - Unrecognized data type <class 'transformers.modeling_outputs.CausalLMOutputWithCrossAttentions'>, cannot be saved in path ..

   如果遇到该警告导致没有数据 dump 下来，请检查模型 py 文件是否正确使用了 torch 模型