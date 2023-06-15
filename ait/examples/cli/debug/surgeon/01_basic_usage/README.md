# Basic Usage


## 介绍
surgeon工具可通过ait命令行形式启动。


## 运行示例

```bash
ait debug surgeon COMMAND [OPTIONS] [REQUIRED]
```

其中<COMMAND>为surgeon执行模式参数，取值为list、evaluate、optimize和extract；[OPTIONS]和[REQUIRED]为可选项和必选项参数，每个子任务下面的可选项和必选项不同。

## 使用流程

surgeon工具建议按照list、evaluate和optimize的顺序执行。如需切分子图，可使用extract命令导出子图。

操作流程如下：

1. 执行**list**命令列举当前支持自动调优的所有知识库。
2. 执行**evaluate**命令搜索可以被指定知识库优化的ONNX模型。
3. 执行**optimize**命令使用指定的知识库来优化指定的ONNX模型。
4. 执行**extract**命令对模型进行子图切分。