- [FAQ](#faq)
  - [1. gcc版本较高时，全量安装完ait后，`ait -h`命令无法正常运行](#1-gcc版本较高时全量安装完ait后ait--h命令无法正常运行)
  - [2 运行ait -h 出现 cann-toolkit包依赖的相关的库的报错：](#2-运行ait--h-出现-cann-toolkit包依赖的相关的库的报错)
  - [3. 推理完成后，将outputs的结果通过`convert_tensor_to_host`从device侧转到host侧的过程中报错](#3-推理完成后将outputs的结果通过convert_tensor_to_host从device侧转到host侧的过程中报错)
  - [4. input文件夹输入有很多的数据，如果选择其中某一部分做输入进行推理。比如 input文件夹中有50000张图片，如果只选择其中100张进行推理](#4-input文件夹输入有很多的数据如果选择其中某一部分做输入进行推理比如-input文件夹中有50000张图片如果只选择其中100张进行推理)
  - [5. 推理工具运行时，会出现aclruntime版本不匹配告警](#5-推理工具运行时会出现aclruntime版本不匹配告警)
  - [6. 推理工具组合输入进行推理时遇到"save out files error"](#6-推理工具组合输入进行推理时遇到save-out-files-error)
  - [7. acl open device 0 failed推理npu设备打开失败](#7-acl-open-device-0-failed推理npu设备打开失败)
  - [8. tensorsize与filesize 不匹配推理失败](#8-tensorsize与filesize-不匹配推理失败)
  - [9. 推理命令执行正常，增加profiler参数使能profiler功能时出现报错，提示推理命令中的路径找不到](#9-推理命令执行正常增加profiler参数使能profiler功能时出现报错提示推理命令中的路径找不到)
  - [10. 使用与npu型号不匹配的om模型进行推理](#10-使用与npu型号不匹配的om模型进行推理)
  - [11. 使用benchmark命令行工具推理和调用benchmark的python接口`ais_bench.infer.interface.InferSession.infer`进行推理时结果不一致。](#11-使用benchmark命令行工具推理和调用benchmark的python接口ais_benchinferinterfaceinfersessioninfer进行推理时结果不一致)

# FAQ
## 1. gcc版本较高时，全量安装完ait后，`ait -h`命令无法正常运行
**故障现象**
全量安装完ait后，`ait -h`命令报错：
```bash
  File "/root/miniconda3/envs/cl_ptq/lib/python3.7/site-packages/pkg_resources/__init__.py", line 2517, in load
    return self.resolve()
  File "/root/miniconda3/envs/cl_ptq/lib/python3.7/site-packages/pkg_resources/__init__.py", line 2523, in resolve
    module = __import__(self.module_name, fromlist=['__name__'], level=0)
  File "/root/miniconda3/envs/cl_ptq/lib/python3.7/site-packages/ais_bench/infer/main_cli.py", line 17, in <module>
    from ais_bench.infer.benchmark_process import benchmark_process
  File "/root/miniconda3/envs/cl_ptq/lib/python3.7/site-packages/ais_bench/infer/benchmark_process.py", line 31, in <module>
    from ais_bench.infer.interface import InferSession, MemorySummary
  File "/root/miniconda3/envs/cl_ptq/lib/python3.7/site-packages/ais_bench/infer/interface.py", line 19, in <module>
    import aclruntime
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /root/miniconda3/envs/cl_ptq/lib/python3.7/site-packages/aclruntime.cpython-37m-aarch64-linux-gnu.so)

```
'gcc --version'命令查看gcc版本：
```bash
gcc --version gcc (conda forge gcc 12.2.0-19) 12.2.0
Copyright (C) 2022 Free Software Foundation, Inc.
This is free software; see the source for copying conditions. T There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
**原因分析**
可能原因1：高版本gcc需要更高版本的libstdc++.so.6的库文件，
可能原因2：编译aclruntime包所依赖的libstdc++.so.6库版本比实际使用的libstdc++.so.6库版本高，一般conda环境会出这种问题
输入命令`strings /lib64/libstdc++.so.6 | grep GLIBCXX`，得到以下结果：
```bash
GLIBC GLIBCXX 3.4 
GLIBCXX 3.4.1 
GLIBCXX 3.4.2 
GLIBCXX 3.4.3 
GLIBCXX 3.4.4 
GLIBCXX 3.4.5 
GLIBCXX 3.4.6 
GLIBCXX 3.4.7 
GLIBCXX 3.4.8 
GLIBCXX 3.4.9 
GLIBCXX 3.4.10 
GLIBCXX 3.4.11 
GLIBCXX 3.4.12 
GLIBCXX 3.4.13 
GLIBCXX 3.4.14 
GLIBCXX 3.4.15 
GLIBCXX 3.4.16 
GLIBCXX 3.4.17 
GLIBCXX 3.4.18 
GLIBCXX 3.4.19 
GLIBCXX 3.4.20 
GLIBCXX 3.4.21 
GLIBCXX 3.4.22 
GLIBCXX 3.4.23 
GLIBCXX 3.4.24 
GLIBC 2.17 
GLIBC 2.18 
GLIBCXX DEBUG MESSAGE LENGTH 
GA+GLIBCXX ASSERTIONS
```
可以发现里面没有`GLIBCXX_3.4.29`（实际缺哪个版本以ImportError报错为准），说明/lib64/libstdc++.so.6(ImportError提示的路径，以ImportError报错为准)这个软链接指向的`libstdc++.so*`文件版本相对于gcc过低，导致了之前的问题。
**解决方法**
- 输入`find / -name "libstdc++.so.6.0.*"`命令，在服务器上全局搜索比ImportError报错的版本更高的libstdc++.so.6库（比如报错GLIBCXX_3.4.29，可以找libstdc++.so.6.0.29级以上版本的库），找到存在libstdc++.so.6.0.29及以上版本的路径。
- 输入`cp <存在更高版本libstdc++.so.6的路径>/<更高版本的libstdc++.so.6库> /lib64/libstdc++.so.6.0.29`,（其中/lib64/libstdc++.so.6.0.29为实际ImportError提示的路径）将更高版本的libstdc++.so.6库复制到ImportError提示的路径。
- 输入`rm /lib64/libstdc++.so.6`，移除原来的软链接。
- 输入`ln -s /lib64/libstdc++.so.6.0.29 /lib64/libstdc++.so.6`，（其中/lib64/libstdc++.so.6.0.29为实际ImportError提示的路径），建立新的软链接到`/lib64/libstdc++.so.6.0.29`
- 输入命令`strings /lib64/libstdc++.so.6 | grep GLIBCXX 3.4.29`,可以发现里面有：
```
GLIBCXX 3.4.29
```
- 输入'ait -h'，可以正常显示help的信息：
```bash
Usage: ait [OPTIONS] COMMAND [ARGS]...

  ait(Ascend Inference Tools), provides one-site debugging and optimization
  toolkit for inference use Ascend Devices

Options:
  -h, --help  Show this message and exit.

Commands:
  analyze    Analyze tool to analyze model support
  benchmark  benchmark tool to get performance data including latency and
             throughput
  convert    Model convert tool to convert offline model
  debug      Debug a wide variety of model issues
  profile    profile tool to get performance datProfiling, as a professional
             performance analysis tool for Ascension AI tasks, covers the
             collection of key data and analysis of performance indicators
             during AI task execution.
  transplt   Transplant tool to analyze inference applications

```

## 2 运行ait -h 出现 cann-toolkit包依赖的相关的库的报错：
**故障现象**
```
[INFO] import ais bench.infer.backends.backend error: No module named 'attrs' 
[INFO] import ais bench.infer.backends.backend_trtexec error: No module named 'attrs'
```
```
File "/home/Lihui/.local/lib/python3.7/site-packages/ais bench/infer/backends/backend.py", line 24, in  @attrs.define AttributeError: module 'attrs' has no attribute 'define' 
```
**原因分析**
cann-toolkit包所依赖的python库没有安装或者更新，必备的库如下：
```
attrs
numpy
decorator
sympy
cffi
pyyaml
pathlib2
psutil
protobuf
scipy
requests
absl-py
```
**解决方法**
- 首先升级pip：
```
pip3 install --upgrade pip
```
- 安装缺失或者版本低的库
```
pip3 install <缺失的库>
```


## 3. 推理完成后，将outputs的结果通过`convert_tensor_to_host`从device侧转到host侧的过程中报错
**故障现象**
推理多输出的模型时出现报错：
```
param2 pointer is nullptr
CheckCopyValid failed. ret=1004
TensorBuffer::TensorBufferCopy failed. ret=1004
Traceback (most recent call last):
  File "/usr/local/python3.7.5/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/usr/local/python3.7.5/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/yanhe13/.local/lib/python3.7/site-packages/ais_bench/__main__.py", line 3, in <module>
    exec(open(os.path.join(cur_path, "infer/__main__.py")).read())
  File "<string>", line 445, in <module>
  File "<string>", line 308, in main
  File "<string>", line 96, in warmup
  File "<string>", line 111, in run_inference
  File "/home/yanhe13/.local/lib/python3.7/site-packages/ais_bench/infer/interface.py", line 97, in run
    self.convert_tensors_to_host(outputs)
  File "/home/yanhe13/.local/lib/python3.7/site-packages/ais_bench/infer/interface.py", line 76, in convert_tensors_to_host
    tensor.to_host()
RuntimeError: [1004][Invalid parameter] 
[1003][Invalid Pointer] Free failed, ptrData is nullptr.[INFO] unload model success, model Id is 1
```
**故障原因**
用netron查看对应模型的onnx模型的MODEL PROPERTIES，以某模型为例，如下所示：
![输入图片说明](https://foruda.gitee.com/images/1686901085203861240/f562b50b_12576095.png "modelpro.png")
可以看到模型的输出中有两个名称重复的输出，这会造成benchmark将两个shared_ptr指针指向同一内存，会重复free，造成了错误。
**解决方法**
使用ait 的 onnx-modifier工具将重复的输出删除，
删除前：
![输入图片说明](https://foruda.gitee.com/images/1686901684490467542/0fbd5f88_12576095.png "重复.png")
删除后：
![输入图片说明](https://foruda.gitee.com/images/1686901707989951309/bb757fa7_12576095.png "不重复.png")
将修改后的模型用atc转换成om模型后使用benchmark推理就能正常运行。

## 4. input文件夹输入有很多的数据，如果选择其中某一部分做输入进行推理。比如 input文件夹中有50000张图片，如果只选择其中100张进行推理

当前推理工具针对input文件夹中数据是全部读取的，没有读取某部分数据功能

如果需要该功能，可以通过如下脚本命令执行，生成某一部分的软链接的文件夹，传入到推理程序中。

```bash
# 首先搜索src目录下的所有的JPEG的文件  然后选取前100个 然后通过软链接的方式链接dst文件夹中
find ./src -type f -name "*.JPEG" | head -n 100 | xargs -i ln -sf {} ./dst
```

## 5. 推理工具运行时，会出现aclruntime版本不匹配告警
**故障现象**

- 故障命令：

```bash
root#  python3 -m ais_bench --model /home/lhb/code/testdata/resnet50/model/pth_resnet50_bs1.om --loop 2
```
- 报错信息：
```
[WARNING] aclruntime version:0.0.1 is lower please update aclruntime follow any one method
[WARNING] 1. visit https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench to install
[WARNING] 2. or run cmd: pip3  install -v --force-reinstall 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend' to install
```
**故障原因**

环境安装低版本aclruntime, 推理工具运行时使用的是高版本的ais_bench

**处理步骤**

更新aclruntime程序包
## 6. 推理工具组合输入进行推理时遇到"save out files error"

总结：

save out files error错误是在推理输出结果进行切分函数中发生的，推理的输出结果会根据输出文件信息进行切分，在切分过程中执行失败。主要的错误原因有如下两个，一种是切分的batch轴不是最高维度0，如故障现象1；另一种是输入文件大小不对应，导致切分失败，

出现该问题请开启debug模式，仔细检查模型输入信息和文件输入信息和shape信息。检查为什么切分失败。

**故障现象1**

```bash
[ERROR] save out files error array shape:(1, 349184, 2) filesinfo:[['prep/2002_07_19_big_img_18.bin', 'prep/2002_07_19_big_img_90.bin', 'prep/  2002_07_19_big_img_130.bin', 'prep/2002_07_19_big_img_135.bin', 'prep/  2002_07_19_big_img_141.bin', 'prep/2002_07_19_big_img_158.bin', 'prep/  2002_07_19_big_img_160.bin', 'prep/2002_07_19_big_img_198.bin', 'prep/  2002_07_19_big_img_209.bin', 'prep/2002_07_19_big_img_230.bin', 'prep/  2002_07_19_big_img_247.bin', 'prep/2002_07_19_big_img_254.bin', 'prep/  2002_07_19_big_img_255.bin', 'prep/2002_07_19_big_img_269.bin', 'prep/  2002_07_19_big_img_278.bin', 'prep/2002_07_19_big_img_300.bin']]  files_count_perbatch:16 ndata.shape0:1
```
**故障原因1**

input文件由16个文件组成，推理输出进行结果文件切分时，默认按shape的第一维切分，而shape最高维度是1，不是16的倍数。所以报错

**处理步骤1**

推理工具参数"--output_batchsize_axis"取值为1。 改成以shape第2维进行切分

**故障现象2**

```
[ERROR] save out files error array shape:(1, 256, 28, 28) filesinfo:[['dump_data_npu/group_in.npy', 'padding_infer_fake_file']] files_count_perbatch:2 ndata.shape0:1
```

**故障原因2**

从错误打印上分析，推理的输入文件有两个，第二个文件padding_infer_fake_file是为了补齐长度的构造的文件，出现这个文件的情况是因为输入文件的大小与模型的对应输入大小不相等，是倍数关系，本例中是2倍的关系，也就说模型输入的大小是输入文件大小的两倍，所以会增加一个构造文件。

本打印的函数是根据模型文件个数切分模型的输出，默认按照输出shape的最高维进行切分，因为最高 维度为1，切分不了2，所以报错了。

**处理步骤2**

多个输入文件进行组batch进行推理主要是针对一些batch大于1的模型。比如batch=2的模型，那么2个输入的文件组合在一起进行推理。但是本样例场景下，输入文件的数据是模型需要的一半大小。所以自动进行组batch，并且增加了一个padding_infer_fake_file文件，但其实是用户的输入文件大小不应该输入一半，输入错了。因为该模型不是多batch的模型。

将输入文件大小修改为模型输入大小。问题解决。

## 7. acl open device 0 failed推理npu设备打开失败
**故障现象**
- 故障命令：
```
python3 -m ais_bench --model ./bert_base_chanese_bs32.om --device 0
```
- 报错信息：
```
[INFO] acl init success
EL003: The argument is invalid.
       ...
	   Failed to open device,retCode=0x7020014,deviceId=0.[FUNC:Init][FILE:device.cc][LINE:211]
	   ...
	   open device 0 failed,runtime result = 507033.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE：162]
[ERROR] acl open device 0 failed
SetDevice failed.ret=507033
```
**原因分析**

acl open device 0 failed一般是因为驱动和固件安装或者硬件有问题，请排查驱动和固件或者硬件，看下是否安装正常**

## 8. tensorsize与filesize 不匹配推理失败
**故障现象**
- 故障命令
```
python3 -m ais_bench --model ./testdata/resnet50/model/pth_resnet50_bs1.om --input ./testdata/resnet50/input/602112/602112.bin
```
- 报错信息
```
[INFO] try get model batchsize:1
[ERROR] arg0 tensorsize: 196608 filesize: 602112 not match
```
**故障原因**

- 出现该问题主要是因为模型输入大小与文件大小不匹配导致。

- 模型pth_resnet50_bs1.om输入actual_input_1为1*3*256*256，合计196608字节大小。但input参数602112.bin大小为602112。与om文件输入要求不匹配。

**处理步骤**

请查看模型输入需要大小。将文件输入文件大小调整为对应大小。

本例中更换input参数对象为196608字节大小的文件  即可解决问题。

## 9. 推理命令执行正常，增加profiler参数使能profiler功能时出现报错，提示推理命令中的路径找不到
**故障现象**

- 基础推理命令执行正常
```
$ python3 -m ais_bench --model=search_bs1.om --output ./
```
- 在基础推理命令上增加profiler参数使能，报错，提示模型路径找不到
```
$ python3 -m ais_bench --model=search_bs1.om --output ./ --profiler 1
```
- 报错信息
```
[ERROR] load model from file failed, model file is search_bs1.om
RuntimeError:[1][ACL:invalid parameter]
```
- 基础命令中执行成功
```
$ python3 -m ais_bench --model=/home/search_bs1.om --output ./  --input ./1.bin,./2.bin
```
- 基础命令中增加profiler参数使能，报错，提示文件输入路径不存在
```
$ python3 -m ais_bench --model=/home/search_bs1.om --output ./  --input ./1.bin,./2.bin --profiler 1
```
- 报错信息
```
[ERROR] Invalid args. ./1.bin of --input is invalid
```
**故障原因**

出现该问题原因是因为使能profiler功能后，相对路径被profiler模块解析时会有些问题，导致运行目录切换，相对路径找不到，当前版本暂未修复。

**处理步骤**

​	出现该问题，请将model  input output等参数里的相对路径修改为绝对路径，这样可以临时规避。
## 10. 使用与npu型号不匹配的om模型进行推理
**故障现象**
用`ait benchmark`命令推理模型，出现以下报错：
```
[INFO] acl init success 
[INFO] open device 0 success 
E19999: Inner Error! 
E19999 index:0 > stream_list.size(): O, check invalid[FUNC:SetStream][FILE:task info.cc][LINE:26] 
TraceBack (most recent call last): 
Task index:0 init failed, ret:-1.[FUNC:InitTaskInfo  FILE:davinci model.cc][LINE:3593] 
[Model][FromData]load model from data failed, ge result[4294967295][FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161] 
[ERROR] load model from file failed, model file is /home/zhangyouling/ait/resnet50.om 
[WARN] Check failed:processModel-LoadModelFromFile(modelPath), ret:1 
[WARN] no model had been loaded, unload failed 
Traceback (most recent call last): 
File "/usr/local/python3.7.5/lib/python3.7/runpy.py", line 193, in _run_module_as main " main_ ", mod spec) 
File "/usr/local/python3.7.5/lib/python3.7/runpy.py", line 85, in _run_code exec(code, run globals) 
File "/home/zhangyouling/.local/lib/python3.7/site-packages/ais bench/_ main_.py", line 18, in  exec(open(os.path.join(cur path, "infer/ _main__ .py")).read( )) 
File "", line 290, in  
File "/home/zhangyouling/.local/lib/python3.7/site-packages/ais _bench/infer/benchmark process.py", line 517, in benchmark process main(args ) 
File "/home/zhangyouling/.local/lib/python3.7/site-packages/ais_bench/infer/benchmark_process.py", line 272, in main session = init inference session(args) 
File "/home/zhangyouling/.Tocal/lib/python3.7/site-packages/ais bench/infer/benchmark_process.py", line 99, in init_inference session session = InferSession(args.device, args.model, acl json path, args.debug, args.loop) 
File "/home/zhangyouling/.local/lib/python3.7/site-packages/ais bench/infer/interface.py", line 77, in _init self.session = aclruntime.InferenceSession(self.model path, self.device id, options ) RuntimeError: [1][ACL: invalid parameter] 
```
**故障原因**
推理采用的om模型是在310P3卡上通过atc或者aoe工具从其他模型转换出来的，当前推理环境的NPU是310卡，导致了上述报错。
**处理方法**
- 在当前环境下重新转换出对应的om模型，用用benchmark推理。
- 或者换一个和npu型号与om模型匹配的环境下用benchmark推理。

## 11. 使用benchmark命令行工具推理和调用benchmark的python接口`ais_bench.infer.interface.InferSession.infer`进行推理时结果不一致。
**故障现象1**
命令行工具和infer接口都能走完推理流程，但是推理后的outputs结果不一致，如下图所示：
![输入图片说明](https://foruda.gitee.com/images/1686907777306767318/2e5f0cb4_12576095.png "20230616-172850(WeLinkPC).png")

**故障现象2**
命令行工具正常能走完推理流程，但是使用infer接口推理报错，报错码如下所示：
```
[INFO] acl init success 
[INFO] open device 0 success 
[INFO] Load model ./output/om/postnet_dyn_Linux_x86_64.om success 
[INFO] create model description success 
[INFO] Load model ./output/om/decoder iter dyn_linux x86_64.om success 
[INFO] create model description success 
[INFO] load model ./output/om/postnet dyn linux x86 64.om success 
[INFO] create model description success Starting run Tacotron2 encoder eecnnn [(1, 184),(1,)] [] 
Traceback (most recent call last): 
File "om_val.py", line 310, in  tacotron2 output, mel lengths = tacotron2.infer tacotron2(seqs, seq_lens, measurements) 
File "om_val.py", line 183, in infer tacotron2 encoder_output = self.encoder.infer([seqs, seq_lens], "dymshape", 3000000) 
File "/root/miniconda3/envs/xzm/lib/python3.7/site-packages/ais_bench/infer/interface.py", line 151, in infer dyshape = "{}:{}".format(indesc[i].name, ",".join(str_shape)) 
IndexError: list index out of range 
```