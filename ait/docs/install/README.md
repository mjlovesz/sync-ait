## 工具安装

### 环境和依赖

- 请参见《[CANN开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/envdeployment/instg/instg_000002.html)》安装昇腾设备开发或运行环境，即toolkit或nnrt软件包。
- 安装python3.7.5。

### 工具安装方式

ait推理工具的安装包括**ait包**和**依赖的组件包**的安装，其中依赖包可以根据需求只添加所需要的组件包。

安装方式包括：**源代码一键式安装**和**按需手动安装不同组件**，用户可以按需选取。

**说明**：

- 安装环境要求网络畅通。
- centos平台默认为gcc 4.8编译器，可能无法安装本工具，建议更新gcc编译器后再安装。
- 安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [昇腾文档](https://www.hiascend.com/zh/document)。安装后用户可通过 **设置CANN_PATH环境变量** ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/nnrt/latest/。若不设置，工具默认会从/usr/local/Ascend/nnrt/latest/和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。
- `TensorFlow` 相关 python 依赖包，参考 [Centos7.6上TensorFlow1.15.0 环境安装](https://bbs.huaweicloud.com/blogs/181055) 安装 TensorFlow1.15.0 环境。(**如不使用TensorFlow模型的精度对比功能则不需要安装**)
- 依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。(**如不使用transplt应用迁移分析功能则不需要安装**)

#### 源代码一键式安装

```shell
git clone https://gitee.com/ascend/ait.git
cd ait/ait

# 添加执行权限
chmod u+x install.sh

# 安装ait，包括debug、profile、benchmark、transplt、analyze等组件
./install.sh

# 重新安装ait及其debug、profile、benchmark、transplt、analyze等组件
./install.sh --force-reinstall

```

#### 按需手动安装不同组件

```shell
git clone https://gitee.com/ascend/ait.git
cd ait/ait

# 添加执行权限
chmod u+x install.sh

# 1. 只安装debug组件
./install.sh --debug

# 2. 只安装benchmark组件
./install.sh --benchmark

# 3. 只安装analyze组件
./install.sh --analyze

# 4. 只安装transplt组件
./install.sh --transplt

# 5. 只安装profile组件
./install.sh --profile
```