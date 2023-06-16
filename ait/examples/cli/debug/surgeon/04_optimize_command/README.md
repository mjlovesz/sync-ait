# Optimize Command

## 介绍
使用指定的知识库来优化指定的ONNX模型。

```bash
ait debug surgeon optimize [OPTIONS] [REQUIRED]
```

optimize可简写为opt。

参数说明：

| 参数       | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 是否必选 |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| -------- |
| OPTIONS  | 额外参数。可取值：<br/>    -k/--knowledges：知识库列表。可指定知识库名称或序号，以英文逗号“,”分隔。默认启用除修复性质以外的所有知识库。<br/>    -bk/--big-kernel：transform类模型大kernel优化的开关，当开关开启时会启用大kernel优化知识库。关于大kernel优化的介绍请参考[示例](../../../examples/cli/debug/surgeon/06_big_kernel_optimize/README.md)<br/>    -as/--attention-start-node：第一个attention结构的起始节点，与-bk配合使用，当启用大kernel优化开关时，需要提供该参数。<br/>    -ae/--attention-end-node：第一个attention结构的结束节点，与-bk配合使用，当启用大kernel优化开关时，需要提供该参数。<br/>    -t/--infer-test：当启用这个选项时，通过对比优化前后的推理速度来决定是否使用某知识库进行调优，保证可调优的模型均为正向调优。启用该选项需要安装额外依赖[inference]，并且需要安装CANN。<br/>    -soc/--soc-version：使用的昇腾芯片版本。默认为Ascend310P3。仅当启用infer-test选项时有意义。<br/>    -d/--device：NPU设备ID。默认为0。仅当启用infer-test选项时有意义。<br/>    --loop：测试推理速度时推理次数。仅当启用infer-test选项时有意义。默认为100。<br/>    --threshold：推理速度提升阈值。仅当知识库的优化带来的提升超过这个值时才使用这个知识库，可以为负，负值表示接受负优化。默认为0，即默认只接受推理性能有提升的优化。仅当启用infer-test选项时有意义。<br/>    --input-shape：静态Shape图输入形状，ATC转换参数，可以省略。仅当启用infer-test选项时有意义。<br/>    --input-shape-range：动态Shape图形状范围，ATC转换参数。仅当启用infer-test选项时有意义。<br/>    --dynamic-shape：动态Shape图推理输入形状，推理用参数。仅当启用infer-test选项时有意义。<br/>    --output-size：动态Shape图推理输出实际size，推理用参数。仅当启用infer-test选项时有意义。<br/>    --help：工具使用帮助信息。 | 否       |
| REQUIRED | -i/--input：输入ONNX待优化模型，必须为.onnx文件。<br/>    -of/--output-file：输出ONNX模型名称，用户自定义，必须为.onnx文件。优化完成后在当前目录生成优化后ONNX模型文件。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 是       |


## 运行示例

```bash
ait debug surgeon optimize --input=aasist_bs1_ori.onnx --output-file=aasist_bs1_ori_out.onnx
```

输出示例如下：

```bash
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO - Optimization success
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO - Applied knowledges:
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO -   KnowledgeConv1d2Conv2d
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO -   KnowledgeMergeConsecutiveSlice
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO -   KnowledgeTransposeLargeInputConv
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO -   KnowledgeTypeCast
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO -   KnowledgeMergeCasts
2023-04-27 14:31:33,378 - auto-optimizer-logger - INFO - Path: aasist_bs1_ori.onnx -> aasist_bs1_ori_out.onnx
```