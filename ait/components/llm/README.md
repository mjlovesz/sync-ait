# 大模型推理精度工具（Large Language Model Debug tool)
## Dump组件
### 安装方式
#### 1. 下载whl包构建
需要下载框架whl和工具whl。
- 下载链接：
    - arm:
    > [框架whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231206/ait-0.0.1-py3-none-linux_aarch64.whl)           
    > [工具whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231206/ait_llm-0.1.0-py3-none-linux_aarch64.whl)
    - x86:
    > [框架whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231206/ait-0.0.1-py3-none-linux_x86_64.whl)            
    > [工具whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231206/ait_llm-0.1.0-py3-none-linux_x86_64.whl)
- 安装方式：
    ```
    # 安装框架whl
    pip3 install ait-0.0.1-py3-none-linux_aarch64.whl
    # 安装工具whl
    pip3 install ait_llm-0.1.0-py3-none-linux_aarch64.whl
    ```
#### 2. 下载源码编译安装
- 需要下载ait仓后编译使用
- 执行命令如下：
```
git clone https://gitee.com/ascend/ait.git
cd ait/ait
chmod +x install.sh
# 如果需要重装可在下面脚本执行添加 --force-reinstall
./install.sh --llm
```
#### 3. 验证是否安装成功
- 执行如下命令：
```
ait llm dump -h
```
如果打屏有相应参数说明即安装成功。
### 使用方式
```
ait llm dump --exec "bash run.sh patches/models/modeling_xxx.py"
```
### 参数说明

| 参数名                      | 描述                                       | 必选   |
| ------------------------ | ---------------------------------------- | ---- |
| -sd，--only-save-desc          | 只保存tensor描述信息开关，默认为否。使用方式：-sd       | 否    |
| -ids，--save-operation-ids | 选择dump指定索引的tensor，默认为空，全量dump。使用方式：-ids 24_1,2_3_5     | 否    |
| -er，--execute-range          | 指定dump的token轮次范围，区间左右全闭，默认为第0次，使用方式：-er 1,3。| 否    |
| -child，--save-operation-child | 选择是否dump所有子操作的tensor数据，使用方式：-child True| 否    |
| -time，--save-time         | 选择保存的时间节点，取值[0,1,2]，0代表保存执行前(before)，1代表保存执行后(after)，2代表前后都保存。使用方式：-time 0  | 否    |
| -opname，--operation-name        | 指定需要dump的算子类型，支持模糊指定，如selfattention只需要填写self。使用方式：-opname self | 否    |
| -tiling，--save-tiling           | 选择是否需要保存tiling数据，默认为false。使用方式：-tiling                | 否    |
| --exec           | 指定拉起执行大模型推理脚本的命令，使用示例： --exec "bash run.sh patches/models/modeling_xxx.py"|是    |
| -o, --output            | 指定dump数据的输出目录，默认为'./'，使用示例：-o aasx/sss | 否    |
