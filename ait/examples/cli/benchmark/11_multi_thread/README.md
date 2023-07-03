# Multi thread


## 介绍
benchmark推理工具目前提供多线程推理功能

## 准备
根据[README.md](https://gitee.com/ascend/ait/blob/master/ait/components/benchmark/backend/concur/README.md)
完成多线程推理工具的编译

## 运行示例
1. 带数据推理场景，输入需要以文件夹的形式。默认情况下，输出信息仅打屏显示。

    ```bash
    ./concur model=pth_resnet50_bs1.om input=./data/
    ```
    其中，model为OM离线模型文件路径，input为输入文件夹路径,以","分割。

2. 调试模式。开启debug调试模式。

    ```bash
    ./concur model=pth_resnet50_bs1.om input=./data/ debug=1
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

3. 保存结果场景。

    ```bash
    ./concur model=pth_resnet50_bs1.om input=./data/ output=./result/
    ```

    其中output为保存文件夹路径，文件夹需要提前创建，结果以平铺的方式存入该路径。

   - 示例

    ```bash
    # 输入的input文件夹内容如下
    ls ./data/
    196608-0.bin  196608-1.bin  196608-2.bin  196608-3.bin  196608-4.bin  196608-5.bin  196608-6.bin  196608-7.bin  196608-8.bin  196608-9.bin
    ```

    ```bash
    # 输出的output文件夹内容如下
    result/
    |-- 196608-0_0.bin
    |-- 196608-1_0.bin
    |-- 196608-2_0.bin
    |-- 196608-3_0.bin
    |-- 196608-4_0.bin
    |-- 196608-5_0.bin
    |-- 196608-6_0.bin
    |-- 196608-7_0.bin
    |-- 196608-8_0.bin
    |-- 196608-9_0.bin
    ```

4. 动态shape场景

    以ATC设置[1\~8,3,200\~300,200\~300]，设置档位1,3,224,224为例，本程序将获取实际模型输入组Batch。

    动态Shape的输出大小通常为0，建议通过outputSize参数设置对应输出的内存大小。

    ```bash
    ./concur model=pth_resnet50_dymshape.om input=./data/ dymShape=actual_input_1:1,3,224,224 outputSize=10000
    ```

5. 自动设置Shape模式（动态Shape模型）。

    动态Shape模型输入数据的Shape可能是不固定的，比如一个输入文件Shape为1,3,224,224 另一个输入文件Shape为 1,3,300,300。若两个文件同时推理，则需要设置两次动态Shape参数，当前不支持该操作。针对该场景，增加auto_set_dymshape_mode模式，可以根据输入文件的Shape信息，自动设置模型的Shape参数。

    ```bash
    ./concur model=pth_resnet50_dymshape.om input=./data/ outputSize=100000 auto_set_dymshape_mode=1
    ```