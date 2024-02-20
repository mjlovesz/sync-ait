# TorchAir 图模式精度比对
- 当前支持 torchair GE dump 数据与 FX dump 数据精度比对
***

## Dump 数据
- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理
  ```py
  import torch, torch_npu, torchair
  from llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```
  输出路径为指定的 `dump_path="dump"`
- **FX 模式 dump 数据** 添加 `get_fx_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理
  ```py
  import torch, torch_npu, torchair
  from llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_fx_dump_config()  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```
  输出路径为当前文件夹下的 `gm_{time stamp}_dump`
## Compare 比对
  - 执行 `ait llm compare --my-path [GE dump data] --golden-path [FX dump data]`，输出比对结果 csv 文件
    ```sh
    ait llm compare --my-path dump --golden-path gm_{time stamp}_dump
    ```
    如果当前 GE dump 路径下包含多个图映射关系 `dynamo_original_graph_xxx.txt` 文件，可通过参数 `--ge-graph-path` 指定具体使用的映射关系文件