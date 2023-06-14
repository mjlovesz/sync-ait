# 工具使用指南

## 介绍

本文介绍应用迁移分析工具，提供NV C++推理应用工程迁移分析以及昇腾API推荐

## 工具安装

### 使用容器

在当前目录下运行以下命令以构建镜像：
```shell
docker build --no-cache -t ait-transplt:latest .
```
运行以下命令以上述镜像启动容器：
```shell
docker run -it ait-transplt:latest
```


### 不使用容器
1. 安装Clang工具

依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。以Ubuntu22.04为例：

```shell
sudo apt-get install libclang-14-dev clang-14
```

依赖[加速库头文件](https://ait-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip)，依赖[API映射表](https://ait-resources.obs.cn-south-1.myhuaweicloud.com/config.zip)，下载后解压至安装目录，例如`/usr/local/site-packages/app_analyze`

加速库头文件和API映射表可及时更新，注意格式。



2. 安装ait工具

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

| 标题          | 说明                |
|:------------|-------------------|
| api         | cpp文件中的三方库API     |
| cuda_en     | 是否cuda使能          |
| location    | api在源文件中的位置       |
| mxBase_API  | 对应的可加速的mxBase API |
| Description | mxBase API的简介     |
| Workload    | 预估迁移人力            |
