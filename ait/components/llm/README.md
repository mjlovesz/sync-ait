# 大模型推理精度工具（Large Language Model Debug Tool)
## 简介
- 大模型推理精度工具（llm）提供对大模型推理的数据落盘（dump）以及精度定位（compare）功能。
- 使用依赖 CANN-toolkit，以及加速库 ATB

### 免责声明

本工具仅供调试和开发之用，不适用于生产环境。使用者需自行承担使用风险，并理解以下内容：

- [x] 仅限调试开发使用：此工具设计用于辅助开发人员进行调试，不适用于生产环境或其他商业用途。对于因误用本工具而导致的数据丢失、损坏，本工具及其开发者不承担责任。

- [x] 数据处理及删除：用户在使用本工具过程中产生的数据（包括但不限于dump的数据）属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防泄露或不必要的信息泄露。

- [x] 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。

- [x] 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。

免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。

在使用本工具之前，请**谨慎阅读并理解以上免责声明的内容**。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## 安装方式(任选其一即可)
### 1. 下载whl包安装
- 需要下载框架whl和工具whl。
- ait 框架 whl:
    | 版本  | 发布日期   | 平台 | CANN 版本 | whl 链接                                                                                                                                         | MD5 校验码                       |
    | ----- | ---------- | ---- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
    | 0.1.0 | 2023/12/13 | arm  | 7.0.0.RC1 | [ait-0.0.1-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_aarch64.whl) | 271051e901bb3513c7a0edbd1e096cb2 |
    | 0.1.0 | 2023/12/13 | x86  | 7.0.0.RC1 | [ait-0.0.1-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_x86_64.whl)   | 9903fa06b9ff76cba667abf0cbc4da50 |

- ait-llm 工具 whl：

    | 版本  | 发布日期   | 平台       | CANN 版本 | whl链接                                                                                                                                                       | MD5 校验码                       |
    | ----- | ---------- | ---------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
    | 0.1.0 | 2023/12/13 | arm, abi=0 | 7.0.0.RC1 | [ait_llm-0.1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_aarch64.whl) | 48215f3ce18881f60beab6fad88ce30a |
    | 0.1.0 | 2023/12/13 | arm, abi=1 | 7.0.0.RC1 | [ait_llm-0.1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_aarch64.whl) | b96e8e7e4786f1abcbec1458ca3ede5d |
    | 0.1.0 | 2023/12/13 | x86, abi=0 | 7.0.0.RC1 | [ait_llm-0.1.0-py3-none-linux_x86.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_x86_64.whl)      | c605e9d50891632a09b21e90403b5b96 |
    | 0.1.0 | 2023/12/13 | x86, abi=1 | 7.0.0.RC1 | [ait_llm-0.1.0-py3-none-linux_x86.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_x86_64.whl)      | ea88611dc4358f51a47f7659a36d5a48 |

- 安装方式：
    ```
    # 安装框架whl
    pip3 install ait-0.0.1-py3-none-linux_aarch64.whl
    # 安装工具whl
    pip3 install ait_llm-0.1.0-py3-none-linux_aarch64.whl
    ```
### 2. 下载源码编译安装
- 需要下载ait仓后编译使用
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
## Dump特性
- 【注意】：加速库数据dump仅支持12/05之后的加速库版本。
### 使用方式
```
ait llm dump --exec "bash run.sh patches/models/modeling_xxx.py"
```
### 参数说明

| 参数名                      | 描述                                       | 必选   |
| ------------------------ | ---------------------------------------- | ---- |
| --exec | 指定拉起执行大模型推理脚本的命令，使用示例： --exec "bash run.sh patches/models/modeling_xxx.py"。**注：命令中不支持重定向字符，如果需要重定向输出，建议将执行命令写入shell脚本，然后启动shell脚本。** | 是 |
| --type          | dump类型，可选范围：['model', 'layer', 'op', 'kernel', 'tensor', 'cpu_profiling']，分别表示保存模型拓扑信息、layer拓扑信息、算子信息、kernel算子信息、tesnor数据、profiling数据。默认为['tensor']。使用方式：--type layer tensor | 否    |
| -sd，--only-save-desc          | 只保存tensor描述信息开关，默认为否。使用方式：-sd       | 否    |
| -ids，--save-operation-ids | 选择dump指定索引的tensor，默认为空，全量dump。使用方式：-ids 24_1,2_3_5     | 否    |
| -er，--execute-range          | 指定dump的token轮次范围，区间左右全闭，可以支持多个区间序列，默认为第0次，使用方式：-er 1,3 或 -er 3,5,7,7（代表区间[3,5],[7,7],也就是第3，4，5，7次token。）| 否    |
| -child，--save-operation-child | 选择是否dump所有子操作的tensor数据，仅使用ids场景下有效，默认为true。使用方式：-child True| 否    |
| -time，--save-time         | 选择保存的时间节点，取值[0,1,2]，0代表保存执行前(before)，1代表保存执行后(after)，2代表前后都保存。默认保存after。使用方式：-time 0  | 否    |
| -opname，--operation-name        | 指定需要dump的算子类型，支持模糊指定，如selfattention只需要填写self。使用方式：-opname self | 否    |
| -tiling，--save-tiling           | 选择是否需要保存tiling数据，默认为false。使用方式：-tiling                | 否    |
| --save-tensor-part, -stp | 指定保存tensor的部分，0为仅intensor，1为仅outtensor，2为全部保存，默认为2。使用示例：-stp 1 |否    |
| -o, --output            | 指定dump数据的输出目录，默认为'./'，使用示例：-o aasx/sss | 否    |

### Dump落盘位置
Dump默认落盘路径在当前目录下的atb_temp目录下，具体路径是`./atb_temp/tensors/{PID}_{TID}`目录下。
如果指定output目录，则会生成在`{OUTPUT_DIR}/atb_temp/tensors/{PID}_{TID}`目录下。

## Compare特性
提供有精度问题的数据与标杆数据之间的比对能力。
### 命令行
```
ait llm compare --golden-path golden_data.bin --my-path my-path.bin
```
#### 参数说明

| 参数名             | 描述                                                         | 是否必选 |
| ------------------ | ------------------------------------------------------------ | -------- |
| --golden-path, -gp | 标杆数据路径，当前仅支持单个数据文件路径，后续将支持文件夹   | 是       |
| --my-path, -mp     | 待比较的数据路径，当前仅支持单个数据文件路径，后续将支持文件夹 | 是       |
| --log-level, -l    | 日志级别，默认为info                                         | 否       |



## FAQ
- **1.命令执行成功，但是没有数据dump下来：**
    - 请先检查加速库版本是否为2023年12月5日之后的版本。
    - 自查abi与所选包的abi是否匹配，请选择正确abi的版本包；

- **2.执行命令时，报错script_path：**
    - 请自查对应的脚本文件是否具有执行权限