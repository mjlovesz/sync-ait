# Basic Usage

## 介绍

Transplt迁移分析工具，提供NV C++推理应用工程迁移分析以及昇腾API推荐。它使用clang等工具分析应用工程源文件中所调用到的NV加速库API、结构体以及枚举信息，并判断在昇腾库上是否有对应的API、结构体以及枚举，然后给出详细分析报告。帮助用户快速将NV C++推理应用工程迁移到昇腾设备上。

## 使用方法

```shell
ait transplt [OPTIONS]
```

OPTIONS参数说明如下

| 参数                | 说明                                                          | 是否必选 |
| ------------------- | ------------------------------------------------------------ | -------- |
| -s, --source | 待迁移分析工程的目录 | 是 |
| -f, --report-type | 输出报告类型，目前支持csv(xlsx)和json两种 | 否 |
| -l, --log-level | 日志打印级别，默认为INFO。可选项为：DEBUG，INFO，WARNING，ERROR | 否 |
| -t, --tools | 构建工具类型，默认为cmake，目前只支持cmake类型 | 否 |
| --help | 显示帮助信息 | 否 |

## 运行示例

```shell
ait transplt -s /workspace/sample
```

```shell
2023-06-06 16:19:46,881 - INFO - scan_api.py[123] - Scan source files...
2023-06-06 16:19:46,882 - INFO - clang_parser.py[355] - Scanning file: /workspace/sample/xxxx.cpp
2023-06-06 16:20:12,004 - INFO - cxx_scanner.py[33] - Total time for scanning cxx files is 25.12237787246704s
2023-06-06 16:20:17,799 - INFO - csv_report.py[46] - Report generated at: /workspace/sample/output.xlsx
2023-06-06 16:20:17,799 - INFO - scan_api.py[113] - **** Project analysis finished <<<
```

输出结果保存在output.xlsx，会记录代码中所有用到的NV加速库API、结构体以及枚举信息和昇腾的支持情况，结果如下：

| AccAPI              | CUDAEnable | Location        | Context(形参 \| 实参 \| 来源代码 \| 来源位置) | AccLib | AscendAPI                | Description                                            | Workload(人/天) | Params(Ascend:Acc) | AccAPILink | AscendAPILink                                                | AscendLib |
| ------------------- | ---------- | --------------- | --------------------------------------------- | ------ | ------------------------ | ------------------------------------------------------ | --------------- | ------------------ | ---------- | ------------------------------------------------------------ | --------- |
| CUVIDDECODECAPS     | TRUE       | xxx.cpp, 203:21 | []                                            | Codec  | hi_vdec_chn_attr         | 定义解码通道属性结构体。                               | 0.2             |                    |            |                                                              |           |
| cuvidGetDecoderCaps | TRUE       | xxx.cpp, 211:5  | [xxx]                                         | Codec  | hi_mpi_vdec_get_chn_attr | 获取视频解码通道属性。                                 | 0.1             |                    |            | https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_03_0403.html |           |
| cuvidCreateDecoder  | TRUE       | xxx.cpp, 362:5  | [xxx]                                         | Codec  | hi_mpi_vdec_create_chn   | 根据设置的通道属性创建解码通道。                       | 0.2             |                    |            | https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_03_0401.html |           |
| CUVIDPICPARAMS      | TRUE       | xxx.cpp, 526:36 | [xxx]                                         | Codec  | hi_vdec_pic_info         | 定义视频原始图像帧结构。                               | 0.2             |                    |            |                                                              |           |
| cuvidDecodePicture  | TRUE       | xxx.cpp, 534:5  | [xxx]                                         | Codec  | hi_mpi_vdec_send_stream  | 解码前，向解码通道发送码流数据及存放解码结果的buffer。 | 0.2             |                    |            |                                                              |           |
| cuCtxPopCurrent     | TRUE       | xxx.cpp, 544:5  | ['CUcontext * pctx \| \| NO_REF \| NO_REF']   | CUDA   |                          |                                                        | 0.1             |                    |            |                                                              |           |

输出数据说明：

| 标题                                          | 说明      |
| -------------- | ---------------------------------------- |
| AccAPI                                        | 三方加速库API |
| CUDAEnable                                    | 是否CUDA |
| Location                                      | 调用三方加速库API的位置 |
| Context(形参 \| 实参 \| 来源代码 \| 来源位置) | 三方加速库API参数及上下文，包括形参、实参、来源代码文件以及来源位置 |
| AccLib                                        | API所属三方加速库 |
| AscendAPI                                     | 推荐的昇腾API |
| Description                                   | API描述 |
| Workload(人/天)                               | 迁移工作量（人/天） |
| Params(Ascend:Acc) | 昇腾API和三方加速库API形参对应关系 |
| AccAPILink | 三方加速库API文档链接 |
| AscendAPILink | 昇腾API文档链接 |
| AscendLib | 推荐的昇腾API所在库 |
