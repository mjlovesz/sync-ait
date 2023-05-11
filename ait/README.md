#  AIT

## 目录
- [介绍](#介绍)
- [工具安装](#工具安装)
- [工具使用](#工具使用)
- [参考](#参考)
- [许可证](#许可证)
- [免责声明](#免责声明)

## 介绍
AIT(Ascend Inference Tools)作为昇腾统一推理工具，提供客户一体化开发工具，支持一站式调试调优，当前包括debug、profile、analyze等组件。

### ATI各子功能介绍
- ait profile benchmark: 用来针对指定的推理模型运行推理程序，并能够测试推理模型的性能（包括吞吐率、时延）。（[快速入门指南](docs/profile/benchmark/README.md)）
- ait debug surgeon: 使能ONNX模型在昇腾芯片的优化，并提供基于ONNX的改图功能。（[快速入门指南](docs/debug/surgeon/README.md)）
- ait debug compare: 提供自动化的推理场景精度比对，用来定位问题算子。（[快速入门指南](docs/debug/compare/README.md)）

## 工具安装

### 环境和依赖

- 请参见《[CANN开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/envdeployment/instg/instg_000002.html)》安装昇腾设备开发或运行环境，即toolkit或nnrt软件包。
- 安装Python3。

### 工具安装方式

ait推理工具的安装包括**ait包**和**依赖的组件包**的安装，其中依赖包可以根据需求只添加所需要的组件包。


**说明**：

- 安装环境要求网络畅通。
- 安装 `python3.7.5` 环境
- centos平台默认为gcc 4.8编译器，可能无法安装本工具，建议更新gcc编译器后再安装。
- 安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [昇腾文档](https://www.hiascend.com/zh/document)。安装后用户可通过设置CANN_PATH环境变量，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/nnrt/latest/。若不设置，工具默认会从/usr/local/Ascend/nnrt/latest/和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。
- **`TensorFlow` 相关 python 依赖包**，参考 [Centos7.6上tensorflow1.15.0 环境安装](https://bbs.huaweicloud.com/blogs/181055) 安装 TensorFlow1.15.0 环境


#### 源代码一键式安装

```shell
git clone https://gitee.com/ascend/ait.git
cd ait

# 安装ait，包括debug、profile组件
pip3 install .[debug,profile] --force-reinstall

# 或者可以安装指定的组件包
pip3 install .[debug] --force-reinstall
pip3 install .[profile] --force-reinstall

```

#### 按需手动安装不同组件

```shell
git clone https://gitee.com/ascend/ait.git
cd ait

# 1. install ait pkg
pip3 install . --force-reinstall

# 2. install compare pkg
cd ait/components/debug/compare
pip3 install . --force-reinstall

# 3. install surgeon pkg
cd ../surgeon
pip3 install . --force-reinstall

# 4. install benchmark pkg
cd ../../profile/benchmark

# 4.1 构建aclruntime包
pip3 wheel ./backend/ -v
# 4.3 构建ais_bench推理程序包
pip3 wheel ./ -v


# 4.2 安装aclruntime
pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
# 4.3 安装ais_bench推理程序
pip3 install ./ais_bench-{version}-py3-none-any.whl
```

## 工具使用

### 命令格式说明

ait工具可通过ait可执行文件方式启动，若安装工具时未提示Python的HATH变量问题，或手动将Python安装可执行文件的目录加入PATH变量，则可以直接使用如下命令格式：

```bash
ait <TASK> <SUB_TASK> [OPT] [ARGS]
```


其中，```<TASK>```为任务类型，当前支持debug、profile，后续可能会新增其他任务类型，可以通过如下方式查看当前支持的任务列表：

```bash

ait -h
Usage: ait [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  analyze
  debug
  profile
```

```<SUB_TASK>```为子任务类型，当前在debug任务下面，有surgeon、compare，在profile任务下面，有benchmark。后续每个任务下面的子任务类型，也会新增，可以通过如下方式查看每个任务支持的子类任务列表：

1、debug任务支持的功能示例：

```bash
ait debug -h
Usage: ait debug [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  compare  one-click network-wide accuracy analysis of gold models.
  surgeon  main entrance of auto optimizer.
```

2、profile任务支持的功能示例：

```bash
ait profile -h
Usage: ait profile [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  benchmark  Inference tool to get performance data including latency and
             throughput
```


```[OPT]```和```[ARGS]```为可选项以及参数，每个任务下面的可选项和参数都不同，以debug任务下面的compare子任务为例，可以通过如下方式获取


```bash
ait debug compare -h
Usage: ait debug compare [OPTIONS]

Options:
  -gm, --golden-model TEXT     <Required> The original model (.onnx or .pb)
                               file path  [required]
  -om, --om-model TEXT         <Required> The offline model (.om) file path
                               [required]
  -i, --input TEXT             <Optional> The input data path of the model.
                               Separate multiple inputs with commas(,). E.g:
                               input_0.bin,input_1.bin
  -c, --cann-path TEXT         <Optional> The CANN installation path
  -o, --output TEXT            <Optional> The output path
  -s, --input-shape TEXT       <Optional> Shape of input shape. Separate
                               multiple nodes with semicolons(;). E.g:
                               input_name1:1,224,224,3;input_name2:3,300
  -d, --device TEXT            <Optional> Input device ID [0, 255], default is
                               0.
  --output-size TEXT           <Optional> The size of output. Separate
                               multiple sizes with commas(,). E.g: 10200,34000
  -n, --output-nodes TEXT      <Optional> Output nodes designated by user.
                               Separate multiple nodes with semicolons(;).
                               E.g: node_name1:0;node_name2:1;node_name3:0
  --advisor                    <Optional> Enable advisor after compare.
  -dr, --dym-shape-range TEXT  <Optional> Dynamic shape range using in dynamic
                               model, using this means ignore input_shape
  --dump STR2BOOL              <Optional> Whether to dump all the operations
                               ouput. Default True.
  --convert STR2BOOL           <Optional> Enable npu dump data conversion from
                               bin to npy after compare.
  -h, --help                   Show this message and exit.

```

### debug任务使用说明

#### 1. compare子任务简单使用示例
```bash
ait debug compare -h
```

更多使用方式和示例请参考：[compare examples](examples/cli/debug/compare/)

#### 2. surgeon子任务使用说明
```bash
ait debug surgeon -h
```

更多使用方式和示例请参考：[surgeon examples](examples/cli/debug/surgeon/)

### profile任务使用说明
#### 1. benchmark子任务使用说明
```bash
ait profile benchmark -h
```

更多使用方式和示例请参考：[benchmark examples](examples/cli/profile/benchmark/)


## 参考

### AIT资源

* [AIT profile benchmark 快速入门指南](docs/profile/benchmark/README.md)
* [AIT debug surgeon 快速入门指南](docs/debug/surgeon/README.md)
* [AIT debug compare 快速入门指南](docs/debug/compare/README.md)


## 许可证

[Apache License 2.0](LICENSE)


## 免责声明

ait仅提供在昇腾设备上的一体化开发工具，支持一站式调试调优，不对其质量或维护负责。
如果您遇到了问题，Gitee/Ascend/ait提交issue，我们将根据您的issue跟踪解决。
衷心感谢您对我们社区的理解和贡献。


