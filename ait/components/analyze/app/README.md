# App Porting

## 安装

依赖LLVM Clang，需安装[Clang工具](https://releases.llvm.org/)。以Ubuntu为例：

```shell
sudo apt-get install libclang-dev clang
```

依赖[LLVM Clang python bindings](https://github.com/llvm/llvm-project/tree/main/clang/bindings/python)，已在requirements中。
```shell
pip3 install -r requirements.txt
```

依赖加速库头文件，下载后解压至`headers/`

依赖API映射表，下载后解压至`config/`。可及时更新，注意格式。

## 配置

配置**common/kit_config.py**中的**lib_clang_path**为`libclang.so`的路径

## 使用

### 1. 文件夹扫描 porting_advisor.py

**python3 porting_advisor.py [-h] [-s source] [-f report-type] [-t tools] [-l log_level]**  

命令示例: python3 porting_advisor.py -s examples/opencv

### 2. 扫描结果
扫描完成后会在被扫描的工程目录下创建`output.xslx`
各页签对应各个文件扫描结果

其中：

api：cpp文件中的三方库API

cuda_en：是否cuda使能

location：api在源文件中的位置

mxBase_API：对应的可加速的mxBase API

Description： mxBase API的简介

Workload： 预估迁移人力