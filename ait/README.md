#  ait

#### 介绍
AIT(Ascend Inference Tools)作为昇腾统一推理工具，提供客户一体化开发工具，支持一站式调试调优，包括debug、profile、anlyze等组件，每个组件都有不同功能。

## 工具安装

### 环境和依赖

- 请参见《[CANN开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/envdeployment/instg/instg_000002.html)》安装昇腾设备开发或运行环境，即toolkit或nnrt软件包。
- 安装Python3。

### 工具安装方式

ait推理工具的安装包括**ait包**和**依赖的组件包(当前支持profile包、debug包)**的安装，其中依赖包可以根据需求只添加所需要的组件包。


**说明**：

- 安装环境要求网络畅通。
- centos平台默认为gcc 4.8编译器，可能无法安装本工具，建议更新gcc编译器后再安装。
- 本工具安装时需要获取CANN版本，用户可通过设置CANN_PATH环境变量，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/nnrt/latest/。若不设置，工具默认会从/usr/local/Ascend/nnrt/latest/和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。


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
