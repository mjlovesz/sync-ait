# ait tensor view功能使用指南

### 简介
tensor-view工具提供了查看tensor的接口，源数据是dump后生成的bin文件。对其进行链式切片、转置操作。默认每次都会打印统计信息和tensor.shape。可以选择打印tensor本身和保存到文件，文件格式可以选择标准torch格式和ATB格式（与dump生成的相同）

暂不支持Windows

### 环境准备

- 安装 `python >= 3.7` 环境
- **安装ait工具**，安装参考文档：[ait工具安装](https://gitee.com/ascend/ait/blob/master/ait/docs/install/README.md)

### 使用方法

- 通过压缩包方式或 git 命令获取本项目
  ```sh
  git clone https://gitee.com/ascend/ait.git
  ```
- 进入 tensor view 目录
  ```sh
  cd ait/ait/components/tensor_view/ait_tensor_view
  ```

- **数据准备**
    - torch参数文件(.bin)路径
- 命令示例
  ```sh
  ait tensor-view --bin intensor0.bin --operations "[1:2, ..., 3::2];(0,2,1,4);[1:3];(2,0,1)" --output tmp_view/output_view.bin
  ```

### 参数说明

| 参数名               | 描述                                                                                                                                                                                                                                         | 必选 |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----|
| --bin, -b         | Tensor参数文件地址                                                                                                                                                                                                                               | 是  |
| --print, -p       | 是否在控制台打印Tensor，默认不打印                                                                                                                                                                                                                       | 否  |
| --operations, -op | Tensor切片和permute操作的字符串，多个操作使用 **;** 分割，切片操作需要使用 **[ ]** 包裹，转置操作需要使用 **( )** 包裹，这些操作字符串会按顺序应用在Tensor上，执行过程中会检查切片、转置字符串是否valid，切片索引是否超出边界，转置字符串是否与维度对应。需要注意的是，切片支持的包括，**标准切片**、**数字索引**、**Ellipsis**，这三种形式在一个扩展切片中可以自由组合，不支持True/False等形式的切片 | 否  |
| --output, -o      | 经过处理的Tensor的存放路径，如果没有指定具体的路径，默认为当前路径                                                                                                                                                                                                       | 否  |
| --atb, -a         | 是否保存为与bin文件相同的atb格式，默认为False                                                                                                                                                                                                               | 否  |
| -h, --help        | 工具使用帮助信息                                                                                                                                                                                                                                   | 否  |

### 使用场景

