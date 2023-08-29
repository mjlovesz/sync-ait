## 工具安装

### tips
ait工具于2023/08/01完成框架重构，需要全量卸载之前环境中安装的ait以及各子工具，再进行安装。

```bash
cd ait/ait

chmod u+x install.sh

./install.sh --uninstall -y

./install.sh
```

### 环境和依赖

- 请参见《[CANN开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/envdeployment/instg/instg_000002.html)》安装昇腾设备开发或运行环境，即toolkit软件包。建议安装CANN商业版6.3.RC1以上版本。
- 请参见《[GCC安装指引](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/envdeployment/instg/instg_000091.html)》安装GCC编译器7.3.0版本。
- Python版本：支持Python3.7.x、Python3.8.x、Python3.9.x(**如使用TensorFlow模型的精度对比功能则需要Python3.7.x版本**)。

### 工具安装方式

ait推理工具的安装包括**ait包**和**依赖的组件包**的安装，其中依赖包可以根据需求只添加所需要的组件包。

安装方式包括：**源代码一键式安装**和**按需手动安装不同组件**，用户可以按需选取。

**说明**：

- 安装环境要求网络畅通。
- centos 7.6平台默认为gcc 4.8编译器，可能无法安装本工具，建议更新gcc编译器后再安装。
- 安装开发运行环境的昇腾 AI 推理相关驱动、固件、CANN 包，参照 [昇腾文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha002/softwareinstall/instg/instg_000002.html)。安装后用户可通过 **设置CANN_PATH环境变量** ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest。若不设置，工具默认会从环境变量ASCEND_TOOLKIT_HOME和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。
- `TensorFlow` 相关 python 依赖包，参考 [Centos7.6上TensorFlow1.15.0 环境安装](https://bbs.huaweicloud.com/blogs/181055) 安装 TensorFlow1.15.0 环境。(**如不使用TensorFlow模型的精度对比功能则不需要安装**)
- `Caffe` 相关 python 依赖包，参考 [Caffe Installation](http://caffe.berkeleyvision.org/installation.html) 安装 Caffe 环境。(**如不使用 Caffe 模型的精度对比功能则不需要安装**)
- 依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。(**如不使用transplt应用迁移分析功能则不需要安装**)
- 如果使用过程中出现`No module named 'acl'`，请检验CANN包环境变量是否正确。
    > 以下是设置CANN包环境变量的通用方法(假设CANN包安装目录为`ACTUAL_CANN_PATH`)：
    >
    > * 执行如下命令：
    ```
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh
    ```
    > * 普通用户下`ACTUAL_CANN_PATH`一般为`$HOME`，root用户下一般为`/usr/local`
    

#### 源代码一键式安装

```shell
git clone https://gitee.com/ascend/ait.git
cd ait/ait

# 1. 添加执行权限
chmod u+x install.sh

# 2. 以下install.sh根据情况选一个执行
# a. 安装ait，包括debug、profile、benchmark、transplt、analyze等组件（不安装clang等系统依赖库，只影响transplt功能）
./install.sh
  
# b. 安装ait，包括debug、profile、benchmark、transplt、analyze等组件（安装clang等系统依赖库，需要提供sudo权限）
./install.sh --full
  
# c. 重新安装ait及其debug、profile、benchmark、transplt、analyze等组件
./install.sh --force-reinstall
```

#### 按需手动安装不同组件

```shell
git clone https://gitee.com/ascend/ait.git
cd ait/ait

# 添加执行权限
chmod u+x install.sh

# 1. 只安装debug组件（使用compare功能、surgen功能的opt命令下面的--infer-test，需要安装benchmark组件）
./install.sh --debug

# 2. 只安装benchmark组件
./install.sh --benchmark

# 3. 只安装analyze组件
./install.sh --analyze

# 4. 只安装transplt组件（不安装transplt组件依赖的clang系统库）
./install.sh --transplt

# 5. 只安装transplt组件（安装transplt组件依赖的clang系统库，需要提供sudo权限,sles系统安装时，需要手动选择'y',然后继续安装）
./install.sh --transplt --full

# 6. 只安装profile组件
./install.sh --profile

# 7. 只安装convert组件
./install.sh --convert
```


#### 卸载
```shell
# 1. 一个个询问式卸载
./install.sh --uninstall

# 2. 不询问式直接全部卸载
./install.sh --uninstall -y

# 3. 单独组件询问式卸载(例如debug组件)
./install.sh --uninstall --debug

# 4. 不询问式单独组件直接卸载(例如debug组件)
./install.sh --uninstall --debug -y
```

### 常见问题 Q&A

参考：[Ait 安装常见问题](https://gitee.com/ascend/ait/wikis/ait%E7%9A%84%E5%AE%89%E8%A3%85%E4%B8%8E%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE/ait%E5%AE%89%E8%A3%85)

