# Basic Usage


## 介绍
benchmark推理工具可以通过ait命令行方式启动模型测试。


## 运行示例
1. 纯推理场景。默认情况下，构造全为0的数据送入模型推理，输出信息仅打屏显示。

    ```bash
    ait benchmark --om-model *.om
    ```
    其中，*为OM离线模型文件名。

2. 调试模式。开启debug调试模式。

    ```bash
    ait benchmark --om-model /home/model/resnet50_v1.om --output ./ --debug 1
    ```

    调试模式开启后会增加更多的打印信息，包括：
   - 模型的输入输出参数信息

     ```bash
     input:
       #0    input_ids  (1, 384)  int32  1536  1536
       #1    input_mask  (1, 384)  int32  1536  1536
       #2    segment_ids  (1, 384)  int32  1536  1536
     output:
       #0    logits:0  (1, 384, 2)  float32  3072  3072
     ```

   - 详细的推理耗时信息

     ```bash
     [DEBUG] model aclExec cost : 2.336000
     ```
   - 模型输入输出等具体操作信息

## FAQ
使用出现问题时，可参考[FAQ](../../../../docs/benchmark/FAQ.md)