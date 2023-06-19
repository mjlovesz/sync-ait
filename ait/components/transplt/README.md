# ait transplt工具使用指南

## 简介

本文介绍应用迁移分析工具，提供NV C++推理应用工程迁移分析以及昇腾API推荐

## 工具安装

### 使用容器方式安装
#### 安装docker

> 以下docker安装指引以x86版本的Ubuntu22.04操作系统为基准，其他系统需要自行修改部分内容。

a) 更新软件包索引，并且安装必要的依赖软件
```shell
sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common lsb-release
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
#### 构建docker镜像
在当前目录下运行以下命令以构建镜像：
```shell
docker build --no-cache -t ait-transplt:latest .
```
运行以下命令以上述镜像启动容器：
```shell
docker run -it ait-transplt:latest
```


### 不使用容器方式安装
#### 安装Clang工具

依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。以Ubuntu22.04为例：

```shell
sudo apt-get install libclang-14-dev clang-14
```

依赖[加速库头文件](https://ait-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip)，依赖[API映射表](https://ait-resources.obs.cn-south-1.myhuaweicloud.com/config.zip)，下载后解压至安装目录，例如`/usr/local/site-packages/app_analyze`

加速库头文件和API映射表可及时更新，注意格式。



#### 安装ait工具

- 工具安装请见 [ait一体化工具使用指南](../../README.md)


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
### Dockerfile构建报错 `ERROR: cannot verify xxx.com's certificate`

可在Dockerfile中每个wget命令后加--no-check-certificate，有安全风险，由用户自行承担。
