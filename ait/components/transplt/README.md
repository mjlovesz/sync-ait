# ait transplt功能使用指南

## 简介

本文介绍应用迁移分析工具，提供NV C++推理应用工程迁移分析以及昇腾API推荐

## 工具安装

ait transplt功能提供了2种安装方式，一种是宿主机安装方式，直接安装在宿主机操作系统中；另一种是容器安装方式。

### 宿主机方式安装

宿主机方式安装目前支持ubuntu22.04、ubuntu20.04、ubuntu18.04、CentOS 7.6、SLES 12.5这些操作系统。用户可以在工程的`<ait_project_root_path>/ait`目录下运行install.sh安装ait transplt功能。

> ait_project_root_path为ait工程的根目录

#### 普通安装

```shell
bash install.sh --transplt
```

在此模式下将只安装ait tranplt功能和依赖的python库，不会安装clang工具，如需安装clang工具，请使用全量安装方式。

#### 全量安装

```shell
bash install.sh --transplt --full
```

在此模式下会安装ait tranplt功能和相应的python库，以及clang工具，安装clang工具时需要提供sudo权限。
其中，sles系统安装时，需要手动选择'y',然后继续安装。

具体请参见 [ait一体化工具使用指南](../../README.md)

### 容器方式安装

容器方式安装目前提供了Ubuntu 20.04的docker镜像。在`<ait_project_root_path>/ait/components/transplt`目录下运行以下命令以构建镜像：
```shell
docker build --no-cache -t ait-transplt:latest .
```
运行以下命令以上述镜像启动容器：
```shell
docker run -it ait-transplt:latest
```

## 工具使用

一站式ait工具使用命令格式说明如下：

```shell
ait transplt [OPTIONS]
```
OPTIONS参数说明如下：

| 参数          | 说明                                  | 是否必选 |
|-------------|-------------------------------------|------|
| -s, --source | 待扫描的工程路径                            | 是    |
| -f, --report-type | 输出报告类型，支持csv（xlsx），json             | 否    |
| --tools     | 构建工具类型，目前支持cmake                    | 否    |
| --log_level | 日志级别，支持INFO（默认），DEBUG，WARNING，ERROR | 否    |

命令示例如下：

```shell
ait transplt -s /data/examples/simple/
```

```shell
2023-05-13 10:30:35,346 - INFO - scan_api.py[123] - Scan source files...
2023-05-13 10:30:35,347 - INFO - clang_parser.py[303] - Scanning file: /data/examples/simple/exmaple_02-03.cpp
2023-05-13 10:30:49,625 - INFO - cxx_scanner.py[46] - Total time for scanning cxx files is 14.278300523757935s
2023-05-13 10:30:49,791 - INFO - json_report.py[50] - Report generated at: /data/examples/simple/output.json
2023-05-13 10:30:49,791 - INFO - scan_api.py[113] - **** Project analysis finished <<<

```

在待扫描的工程目录下输出output.xlsx，会呈现工程中每个支持的加速库API的信息和支持情况，结果如下：

输出数据说明：

| 标题                                          | 说明      |
| -------------- | ---------------------------------------- |
| AccAPI                                        | 三方加速库API |
| CUDAEnable                                    | 是否CUDA |
| Location                                      | 调用三方加速库API的位置 |
| Context(形参 \| 实参 \| 来源代码 \| 来源位置) | 三方加速库API参数及上下文，包括形参、实参、来源代码文件以及来源位置 |
| AccLib                                        | API所属三方加速库 |
| AscendAPI                                     | 推荐的昇腾API |
| Description                                   | API描述 |
| Workload(人/天)                               | 迁移工作量（人/天） |
| AccAPILink | 三方加速库API文档链接 |
| AscendAPILink | 昇腾API文档链接 |
| AscendLib | 推荐的昇腾API所在库 |

## FAQ
### 如何安装Docker

如果操作系统中没有安装docker，可以参考如下步骤手动进行安装。

> 以下docker安装指引以x86版本的Ubuntu22.04操作系统为基准，其他系统需要自行修改部分内容。
a) 更新软件包索引，并且安装必要的依赖软件
```shell
sudo apt update
sudo apt install apt-transport-https ca-certificates curl wget gnupg-agent software-properties-common lsb-release
```
b) 导入docker源仓库的 GPG key
```shell
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

> 注意：如果当前机器采用proxy方式联网，上面的命令有可能会遇到```curl: (60) SSL certificate problem: self signed certificate in certificate chain``` 的报错问题。遇到这种情况，可以在将curl的运行参数从```curl -fsSL```修改成```curl -fsSL -k```。需要注意的是，这会跳过检查目标网站的证书信息，有一定的安全风险，用户需要谨慎使用并自行承担后果。

c) 将 Docker APT 软件源添加到系统
```shell
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

> 注意：如果上面的命令运行失败了，用户也可以采用如下命令手动将docker apt源添加到系统
>
> ```shell
> sudo echo "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" >> /etc/apt/sources.list
> ```

d) 安装docker
```shell
sudo apt install docker-ce docker-ce-cli containerd.io
```
如果想安装指定版本的docker，可以在上面的命令中添加docker版本信息，如下所示
```shell
sudo apt install docker-ce=<VERSION> docker-ce-cli=<VERSION> containerd.io
```
e) 启动docker服务
一旦安装完成，Docker 服务将会自动启动，可以输入下面的命令查看docker服务的状态
```shell
sudo systemctl status docker
```
如果docker服务没有启动，可以尝试手动启动docker服务
```shell
sudo systemctl start docker
```
f) 以非root用户运行docker命令

默认情况下，只有 root 或者 有 sudo 权限的用户可以执行 Docker 命令。如果想要以非 root 用户执行 Docker 命令，则需要将你的用户添加到 Docker 用户组，如下所示：
```shell
sudo usermod -aG docker $USER
```
其中$USER代表当前用户。

### Dockerfile构建时报错 `ERROR: cannot verify xxx.com's certificate`
可在Dockerfile中每个wget命令后加--no-check-certificate，有安全风险，由用户自行承担。

### 如何下载/更新加速库头文件和API映射表
ait transplt功能依赖[加速库头文件](https://ait-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip)和[API映射表](https://ait-resources.obs.cn-south-1.myhuaweicloud.com/config.zip)，这两者会不定时更新。如果用户需要手动下载或者更新这些文件，可以从对应链接下载后解压至ait transplt工具安装目录。这个安装目录根据您的python3安装位置不同会有不同的值，例如您的python3.7在`/usr/local/bin/python3.7`，那么可以下载后解压至```/usr/local/lib/python3.7/dist-packages/app_analyze```目录。

您可以使用`python3 -c "import app_analyze; print(app_analyze.__path__[0])"`命令来确定具体的安装目录。

您也可以使用如下命令一键式下载并解压到安装目录：

```shell
cd $(python3 -c "import app_analyze; print(app_analyze.__path__[0])") \
    && wget -O config.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/config.zip \
    && unzip config.zip \
    && rm config.zip \
    && wget -O headers.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip \
    && unzip headers.zip \
    && rm headers.zip
```

### 如何手动安装Clang工具
ait transplt功能依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。通常情况下安装脚本会自动进行安装，如果用户有需求手动安装Clang工具，可以参考下面的方法在不同OS中进行安装：
> 为了加速安装过程，本文档尽量使用系统安装工具支持的clang版本。如果用户需要通过源码方式安装Clang，可以参考[这一节](##### 源码编译安装Clang)

##### 在Ubuntu 22.04中安装Clang
```shell
sudo apt-get install libclang-14-dev clang-14
```

##### 在Ubuntu 18.04中安装Clang
```shell
sudo apt-get install libclang-10-dev clang-10
```
> **提示**：如果transplt安装目录下`common/kit_config.py`中的LIB_CLANG_PATH`自动配置失败，则需手动修改，libclang.so一般位于
> `/usr/lib/x86-linux-gnu/libclang-10.so`。

##### 在CentOS 7.6中安装Clang
```shell
yum install centos-release-scl-rh
yum install llvm-toolset-7.0-clang
# 使Clang在当前Session生效
source /opt/rh/llvm-toolset-7.0/enable
# 可选，修改.bashrc便于Clang自动生效
echo "source /opt/rh/llvm-toolset-7.0/enable" >> ~/.bashrc
```
配置环境变量。为防止后续Clang无法自动找到头文件，建议添加如下环境变量。
```shell
export CPLUS_INCLUDE_PATH=/opt/rh/llvm-toolset-7.0/root/usr/lib64/clang/7.0.1/include
```
> **提示**：如果transplt安装目录下`common/kit_config.py`中的LIB_CLANG_PATH`自动配置失败，则需手动修改，libclang.so一般位于
> `/opt/rh/llvm-toolset-7.0/root/usr/lib64/libclang.so.7`。

##### 在SLES 12.5中安装Clang
```shell
sudo zypper install libclang7 clang7-devel
```
配置环境变量。为防止后续Clang无法自动找到头文件，建议添加如下环境变量。
```shell
export CPLUS_INCLUDE_PATH=/usr/lib64/clang/7.0.1/include
```
> **提示**：如果transplt安装目录下`common/kit_config.py`中的LIB_CLANG_PATH`自动配置失败，则需手动修改，libclang.so一般位于
> `/usr/lib64/libclang.so`，可用`sudo find / -name "libclang.so"`命令查找。

##### 源码编译安装Clang
如果无法通过上述方法或者包管理工具安装Clang>=6.0.0，可以在[LLVM Release](https://github.com/llvm/llvm-project/releases)页面尝试下载对应平台的安装包。如果以上方法都不可行，则可以通过源码编译安装LLVM和Clang，详细安装指导参考[Getting Started with the LLVM System](https://llvm.org/docs/GettingStarted.html)。编译LLVM依赖一些软件包，需用户提前确保依赖满足，或者自行手动安装依赖。下表列出了这些依赖，“包名”列是LLVM所依赖的软件包通常的名称，“版本”列是“可以工作“的软件包版本，“说明”列描述了LLVM如何使用这个软件包。

| 包名                                              | 版本         | 说明                   |
| :------------------------------------------------ | :----------- | :--------------------- |
| [CMake](http://cmake.org/)                        | >=3.20.0     | 生成Makefile/workspace |
| [GCC](http://gcc.gnu.org/)                        | >=7.1.0      | C/C++编译器            |
| [zlib](http://zlib.net/)                          | >=1.2.3.4    | 压缩/解压功能          |
| [GNU Make](http://savannah.gnu.org/projects/make) | 3.79, 3.79.1 | 编译Makefile/build     |

下面以Clang7.0.0为例编译安装LLVM和Clang：

a) **获取源码**：通过Git获取源码，包括LLVM和Clang子工程，切换到对应版本。
```shell
git clone https://github.com/llvm/llvm-project.git
git checkout llvmorg-7.0.0
```
或者直接下载对应版本的源码zip包。
```shell
wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-7.0.0.zip
# 如果没有安装wget，可以采用curl
curl -o llvmorg-7.0.0.zip https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-7.0.0.zip
# 解压得到llvm-project-llvmorg-7.0.0目录
unzip -q llvmorg-7.0.0.zip
```
b) **编译和安装LLVM和Clang**：
```shell
cd llvm-project-llvmorg-7.0.0/; mkdir build; cd build
# 建议不开启libcxx;libcxxabi，使用默认的gcc/g++配套的libstdc++
cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -G "Unix Makefiles" ../llvm
make -j32  # 将32换成小于所在机器CPU线程数的数字，或者去除数字，自动设定
make install  # 安装到默认位置/usr/local/lib/
```
c) **配置环境变量**：为防止后续Clang无法自动找到头文件，建议添加如下环境变量。
```shell
export CPLUS_INCLUDE_PATH=/usr/local/lib/clang/7.0.0/include
```
> **提示**：如果transplt安装目录下`common/kit_config.py`中的LIB_CLANG_PATH`自动配置失败，则需手动修改，libclang.so一般位于
> `/usr/local/lib/libclang.so`。