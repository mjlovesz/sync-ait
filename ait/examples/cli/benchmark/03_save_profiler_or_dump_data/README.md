# Save Profiler or Dump Data


## 介绍

profiler，采集性能数据；dump，保存全部算子输出。支持以--acl-json-path、--profiler、--dump参数形式实现。

## 运行示例

+ acl-json-path参数指定acl.json文件，可以在该文件中对应的profiler或dump参数。示例代码如下：

  + profiler

    ```bash
    {
    "profiler": {
                  "switch": "on",
                  "output": "./result/profiler"
                }
    }
    ```

    更多性能参数配置请参见《[CANN 开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/auxiliarydevtool_0002.html)》中的“性能分析工具>高级功能>性能数据采集（acl.json配置文件方式）”章节。

  + dump

    ```bash
    {
        "dump": {
            "dump_list": [
                {
                    "model_name": "{model_name}"
                }
            ],
            "dump_mode": "output",
            "dump_path": "./result/dump"
        }
    }
    ```

    更多dump配置请参见《[CANN 开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/auxiliarydevtool_0002.html)》中的“精度比对工具>比对数据准备>推理场景数据准备>准备离线模型dump数据文件”章节。

  通过该方式进行Profiler采集时，输出的性能数据文件需要参见《[CANN 开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/auxiliarydevtool_0002.html)》中的“性能分析工具>高级功能>数据解析与导出”章节，将性能数据解析并导出为可视化的timeline和summary文件。

+ profiler为固化到程序中的一组性能数据采集配置，生成的性能数据保存在--output参数指定的目录下的profiler文件夹内。

    该参数是通过调用ait/profiler/benchmark/infer/benchmark_process.py中的msprof_run_profiling函数来拉起msprof命令进行性能数据采集的。若需要修改性能数据采集参数，可根据实际情况修改msprof_run_profiling函数中的msprof_cmd参数。示例如下：

    ```bash
    msprof_cmd="{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on".format(
            msprof_bin, args.output, cmd)
    ```

    该方式进行性能数据采集时，首先检查是否存在msprof命令：

    - 若命令存在，则使用该命令进行性能数据采集、解析并导出为可视化的timeline和summary文件。
    - 若命令不存在，则调用acl.json文件进行性能数据采集。
    - 若环境配置了GE_PROFILING_TO_STD_OUT=1，则使用--profiler参数采集性能数据时调用的是acl.json文件。

    msprof命令不存在或环境配置了GE_PROFILING_TO_STD_OUT=1情况下，采集的性能数据文件未自动解析，需要参见《[CANN 开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/auxiliarydevtool_0002.html)》中的“性能分析工具>高级功能>数据解析与导出”章节，将性能数据解析并导出为可视化的timeline和summary文件。

    更多性能数据采集参数介绍请参见《[CANN 开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/auxiliarydevtool_0002.html)》中的“性能分析工具>高级功能>性能数据采集（msprof命令行方式）”章节。

  + acl-json-path优先级高于profiler和dump，同时设置时以acl-json-path为准。

  + profiler参数和dump参数，必须要增加output参数，指示输出路径。

  + profiler和dump可以分别使用，但不能同时启用。

  示例命令如下：
  
  ```bash
  ait benchmark --om-model ./resnet50_v1_bs1_fp32.om --acl-json-path ./acl.json
  ait benchmark --om-model /home/model/resnet50_v1.om --output ./ --dump 1
  ait benchmark --om-model /home/model/resnet50_v1.om --output ./ --profiler 1
  ```