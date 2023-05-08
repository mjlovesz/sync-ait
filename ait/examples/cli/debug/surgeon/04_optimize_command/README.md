# Optimize Command

## 介绍
使用指定的知识库来优化指定的ONNX模型。

```bash
ait debug compare optimize [OPTIONS] INPUT_MODEL OUTPUT_MODEL
```

optimize可简写为opt。

参数说明：

| 参数         | 说明                                                         | 是否必选 |
| ------------ | ------------------------------------------------------------ | -------- |
| OPTIONS      | 额外参数。可取值：<br/>    -k/--knowledges：知识库列表。可指定知识库名称或序号，以英文逗号“,”分隔。默认启用除修复性质以外的所有知识库。<br/>    -t/--infer-test：当启用这个选项时，通过对比优化前后的推理速度来决定是否使用某知识库进行调优，保证可调优的模型均为正向调优。启用该选项需要安装额外依赖[inference]，并且需要安装CANN。<br/>    -s/--soc：使用的昇腾芯片版本。默认为Ascend310P3。仅当启用infer-test选项时有意义。<br/>    -d/--device：NPU设备ID。默认为0。仅当启用infer-test选项时有意义。<br/>    -l/--loop：测试推理速度时推理次数。仅当启用infer-test选项时有意义。默认为100。<br/>    --threshold：推理速度提升阈值。仅当知识库的优化带来的提升超过这个值时才使用这个知识库，可以为负，负值表示接受负优化。默认为0，即默认只接受推理性能有提升的优化。仅当启用infer-test选项时有意义。<br/>    --input-shape：静态Shape图输入形状，ATC转换参数，可以省略。仅当启用infer-test选项时有意义。<br/>    --input-shape-range：动态Shape图形状范围，ATC转换参数。仅当启用infer-test选项时有意义。<br/>    --dynamic-shape：动态Shape图推理输入形状，推理用参数。仅当启用infer-test选项时有意义。<br/>    --output-size：动态Shape图推理输出实际size，推理用参数。仅当启用infer-test选项时有意义。<br/>    --help：工具使用帮助信息。 | 否       |
| INPUT_MODEL  | 输入ONNX待优化模型，必须为.onnx文件。                        | 是       |
| OUTPUT_MODEL | 输出ONNX模型名称，用户自定义，必须为.onnx文件。优化完成后在当前目录生成优化后ONNX模型文件。 | 是       |


## 运行示例

```bash
ait debug compare optimize aasist_bs1_ori.onnx aasist_bs1_ori_out.onnx
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