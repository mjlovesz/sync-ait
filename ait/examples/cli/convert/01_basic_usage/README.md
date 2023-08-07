# Basic Usage

## 介绍

Convert模型转换工具依托AIE（Ascend Inference Engine）推理引擎，提供由ONNX模型转换至om模型的功能。

## 使用场景约束
1. 当前仅支持**Ascend310**以及**Ascend310P**平台的AIE转换；
2. 当前仅支持**FP16**精度下的模型转换
3. 当前已验证模型：Resnet50、DBNet、CRNN

## 运行示例

```shell
ait convert aie --golden-model resnet50.onnx --output-file resnet50.om --soc-version Ascend310P3
```

结果输出如下：
```shell
[INFO] Execute command:['sh', 'build.sh', '-p', '/usr/bin/python3']
[INFO] b'-- The C compiler identification is GNU 11.3.0'
[INFO] b'-- The Cxx compiler identification is GNU 11.3.0'
[INFO] b'-- Detecting C compiler ABI info'
[INFO] b'-- Detecting C compiler ABI info - done'
[INFO] b'-- Checking for working C compiler: /usr/bin/cc - skipped'
[INFO] b'-- Detecting C compiler features'
[INFO] b'-- Detecting C compiler features - done'
[INFO] b'-- Detecting C compiler ABI info'
[INFO] b'-- Detecting C compiler ABI info - done'
[INFO] b'-- Checking for working CXX compiler: /usr/bin/c++ - skipped'
[INFO] b'-- Detecting CXX compiler features'
[INFO] b'-- Detecting CXX compiler features - done'
[INFO] b'-- Configuring done'
[INFO] b'-- Generating done'
[INFO] b'-- Building files have been written to: /xxx/ait/components/convert/model_convert/cpp/build'
[INFO] b'Scanning dependencies of target ait_convert'
[INFO] b'[ 50%] Building CXX object CMakeFiles/ait_convert.dir/aie_convert.cpp.o'
[INFO] b'[100%] Linking CXX executable ait_convert'
[INFO] b'[100%] Built target ait_convert'
[INFO] Run command line: ['sh', 'build.sh', '-p', '/usr/bin/python3']
[INFO] Execute command:['./ait_convert', 'resnet50.onnx', 'resnet50.om', 'Ascend310']
[INFO] b'AIE Model Convert:1'
[INFO] Execute command:['cp', 'resnet50.om', '/xxx/ait']
[INFO] AIE model convert finished, the command: ['./ait_convert', 'resnet50.onnx', 'resnet50.om', 'Ascend310']
[INFO] convert model finished.
```
