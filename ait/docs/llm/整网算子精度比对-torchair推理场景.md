
# 2. 基于torch图模式（torchair）推理场景

## 2.1 GE dump 数据与 FX dump 数据精度比对

### 1）Dump 数据

- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

- **FX 模式 dump 数据** 添加 `get_fx_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_fx_dump_config()  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径为当前文件夹下的 `gm_{time stamp}_dump`

### 2）Compare 精度比对

  - 执行 `ait llm compare --my-path [GE dump data] --golden-path [FX dump data]`，输出比对结果 csv 文件

    ```sh
    ait llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path gm_{time stamp}_dump
    ```

***

## 2.2 融合的 GE dump 数据与关闭融合的 GE dump 数据精度比对

### 1）Dump 数据

- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

- **GE 模式关闭融合 dump 数据** 添加 `get_ge_dump_config`，指定 `fusion_switch_file` 文件，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理

  ```py
  import torch, torch_npu, torchair
  from ait_llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump", fusion_switch_file="fusion_switch.json")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```

  参考的 `fusion_switch.json` 文件

  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```

  输出路径为指定的 `{dump_path}/dump_{time_stamp}`

### 2）Compare 比对

- 执行 `ait llm compare --my-path [GE dump data] --golden-path [fusion off GE dump data]`，输出比对结果 csv 文件

  ```sh
  ait llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path {dump_path}/dump_{time_stamp}
  ```