# 大模型推理精度工具（Large Language Model Debug Tool)

## 简介

- 大模型推理精度工具（llm）提供对大模型推理的数据落盘（dump）以及精度定位（compare）功能。
- 使用依赖 CANN-toolkit，以及加速库 ATB，其中 CANN-toolkit 版本要求参照具体安装方式说明
- 【注意】：加速库数据dump仅支持12/05之后的加速库版本。

## 免责声明

- 本工具仅供调试和开发之用，不适用于生产环境。使用者需自行承担使用风险，并理解以下内容：

  - [X] 仅限调试开发使用：此工具设计用于辅助开发人员进行调试，不适用于生产环境或其他商业用途。对于因误用本工具而导致的数据丢失、损坏，本工具及其开发者不承担责任。
  - [X] 数据处理及删除：用户在使用本工具过程中产生的数据（包括但不限于dump的数据）属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防泄露或不必要的信息泄露。
  - [X] 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。
  - [X] 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。
- 在使用本工具之前，请**谨慎阅读并理解以上免责声明的内容**。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## 安装方式(任选其一即可)

### 1. 下载whl包安装

- 需要下载安装框架 whl 和工具 whl。
- ait 框架 whl:

  | 版本  | 发布日期   | 平台 | CANN 版本 | whl 链接                                                                                                                                      | MD5 校验码                       |
  | ----- | ---------- | ---- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
  | 0.0.1 | 2023/12/13 | arm  | 7.0.0.RC1 | [ait-0.0.1-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_aarch64.whl) | 271051e901bb3513c7a0edbd1e096cb2 |
  | 0.0.1 | 2023/12/13 | x86  | 7.0.0.RC1 | [ait-0.0.1-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_x86_64.whl)   | 9903fa06b9ff76cba667abf0cbc4da50 |
- ait-llm 工具 whl：

  | 版本  | 发布日期   | 平台       | CANN 版本    | whl链接                                                      | MD5 校验码                       | 使用指导                                                     |
  | ----- | ---------- | ---------- | ------------ | ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
  | 1.1   | 2024/05/08 | arm        | 8.0.RC2 B010 | [ait_llm-1.1-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240508/ait_llm-1.1-py3-none-linux_aarch64.whl)                    |0133c8fda39ba78c2b02354b4bcf089c                                | [大模型推理精度工具](../../docs/llm/README.md) |
  | 1.1   | 2024/05/08 | x86        | 8.0.RC2 B010 | [ait_llm-1.1-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240508/ait_llm-1.1-py3-none-linux_x86_64.whl)                     |d453b4b608b4400d77bbfb1b5c702bee                                | [大模型推理精度工具](../../docs/llm/README.md) |
  | 1.0   | 2024/03/22 | arm        | 8.0.RC1      | [ait_llm-1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240325/ait_llm-1.0-py3-none-linux_aarch64.whl)                          |9f7f69d49e017f98006b8191f3951868                                  | [大模型推理精度工具说明文档](../../docs/llm/v1.0/大模型推理精度工具说明文档.md) |
  | 1.0   | 2024/03/22 | x86        | 8.0.RC1      |[ait_llm-1.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240325/ait_llm-1.0-py3-none-linux_x86_64.whl)                          |5a6735c9f04d3938a6384c460399ff9a                                  | [大模型推理精度工具说明文档](../../docs/llm/v1.0/大模型推理精度工具说明文档.md) |
  |       |            |            |              |                                                              |                                  |                                                              |
  | 0.2.1 | 2024/02/08 | arm        | 8.0.RC1.B020 | [ait_llm-0.2.1-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240208/ait_llm-0.2.1-py3-none-linux_aarch64.whl) | 1f24783f0815dbca36e8e787a8bfcf09 | [llm大模型推理精度工具功能说明_v0.2.1](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.2.1.md) |
  | 0.2.1 | 2024/02/08 | x86        | 8.0.RC1.B020 | [ait_llm-0.2.1-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240208/ait_llm-0.2.1-py3-none-linux_x86_64.whl) | 679fae6a5b6ea1f4a749b9554f3e5c37 | [llm大模型推理精度工具功能说明_v0.2.1](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.2.1.md) |
  |       |            |            |              |                                                              |                                  |                                                              |
  | 0.2.0 | 2024/01/17 | arm        | 8.0.RC1      | [ait_llm-0.2.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240117/ait_llm-0.2.0-py3-none-linux_aarch64.whl) | 99b94bf7edd57b63a6e23b987d24f364 | [llm大模型推理精度工具功能说明_v0.2.0](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.2.0.md) |
  | 0.2.0 | 2024/01/17 | x86        | 8.0.RC1      | [ait_llm-0.2.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240117/ait_llm-0.2.0-py3-none-linux_x86_64.whl) | dec5757afedfea8848c5db1bfad3d76c | [llm大模型推理精度工具功能说明_v0.2.0](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.2.0.md) |
  |       |            |            |              |                                                              |                                  |                                                              |
  | 0.1.0 | 2023/12/13 | arm, abi=0 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_aarch64.whl) | 48215f3ce18881f60beab6fad88ce30a | [llm大模型推理精度工具功能说明_v0.1.0](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
  | 0.1.0 | 2023/12/13 | arm, abi=1 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_aarch64.whl) | b96e8e7e4786f1abcbec1458ca3ede5d | [llm大模型推理精度工具功能说明_v0.1.0](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
  | 0.1.0 | 2023/12/13 | x86, abi=0 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_x86_64.whl) | c605e9d50891632a09b21e90403b5b96 | [llm大模型推理精度工具功能说明_v0.1.0](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
  | 0.1.0 | 2023/12/13 | x86, abi=1 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_x86_64.whl) | ea88611dc4358f51a47f7659a36d5a48 | [llm大模型推理精度工具功能说明_v0.1.0](../../docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
- 校验whl包是否正确

  ```
  # 校验whl包是否正确
  md5sum xxxx.whl
  ```

  比对 md5 值与所提供的校验值一致
- 安装方式：

  ```
  # 安装所需版本的框架 whl
  pip3 install ait-0.0.1-py3-none-linux_aarch64.whl
  # 安装所需版本的工具 whl
  pip3 install ait_llm-0.2.0-py3-none-linux_aarch64.whl
  ```

### 2. 下载源码编译安装

- 需要下载ait仓后编译使用
- 对应 CANN-toolkit 版本为 `8.0.RC1`
- 执行命令如下：
  ```
  git clone https://gitee.com/ascend/ait.git
  cd ait/ait
  chmod +x install.sh
  # 如果需要重装可在下面脚本执行添加 --force-reinstall
  ./install.sh --llm
  ```

### 验证是否安装成功

- 执行如下命令：

  ```
  ait llm -h
  ```

  如果打屏有相应参数说明即安装成功。

---

## FAQ

- **1.命令执行成功，但是没有数据dump下来：**

  - 请先检查加速库版本是否为2023年12月5日之后的版本。
  - 自查abi与所选包的abi是否匹配，请选择正确abi的版本包。
  - check下是否设置环境变量ATB_SAVE_TENSOR_RUNNER，若设置，执行unset ATB_SAVE_TENSOR_RUNNER。
  - 开启加速库日志，查看日志中是否有报错，确认所有算子成功运行。
- **2.执行命令时，报错script_path：**

  - 请自查对应的脚本文件是否具有执行权限
