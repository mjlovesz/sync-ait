# porting acl

## 安装

依赖LLVM clang，需安装[clang工具](https://releases.llvm.org/)。以Ubuntu为例：

```shell
apt-get install libclang-dev
sudo apt-get install --fix-missing clang
```

依赖[LLVM clang python bindings](https://github.com/llvm/llvm-project/tree/main/clang/bindings/python)。按装方式有两种：

- [clang(mirrored LLVM python bindings)](https://pypi.org/project/clang/)：依赖系统库，如`libclang-6.0.so。
- [libclang(mirrored LLVM python bindings)](https://readthedocs.org/projects/libclang/)：自带`libclang.so`。

## 使用

API映射表位于`config/DVPP_API_MAP.xlsx`。可及时更新，注意格式。

### 1. 文件夹扫描 porting_advisor.py

**porting-advisor [-h] [-s source] [-f report-type] [-t tools] [-l log_level] **

cxx源文件扫描，支持两种模式：clang-python和clang-tool，在**common/kit_config.py**中配置变量**cxx_scanner_type**。

eg: python porting-advisor -S examples/opencv

### 2. 单文件cxx扫描, 两种方式: clang-python 和 clang-tool, 位于scan目录下

**clang-python**: 如果找不到libclang, 需要配置**common/kit_config.py**中的**lib_clang_path**
**clang-tool**: 如果找不到opencv, 需要配置**common/kit_config.py**中的**opencv_include_path**
