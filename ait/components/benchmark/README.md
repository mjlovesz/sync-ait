

# ais_bench推理工具使用指南

## 简介
本文介绍ais_bench推理工具，用来针对指定的推理模型运行推理程序，并能够测试推理模型的性能（包括吞吐率、时延）。

## 工具安装

### 环境和依赖

- 请参见《[CANN开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/envdeployment/instg/instg_000002.html)》安装昇腾设备开发或运行环境，即toolkit或nnrt软件包。
- 安装Python3、Python包模块numpy、tqdm、wheel。

### 工具安装方式

ais_bench推理工具的安装包括**aclruntime包**和**ais_bench推理程序包**的安装。
安装方式包括：下载whl包安装、一键式编译安装和源代码编译安装。

**说明**：

- 安装环境要求网络畅通。
- centos平台默认为gcc 4.8编译器，可能无法安装本工具，建议更新gcc编译器后再安装。
- 本工具安装时需要获取CANN版本，用户可通过设置CANN_PATH环境变量，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/nnrt/latest/。若不设置，工具默认会从/usr/local/Ascend/nnrt/latest/和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。

#### 下载whl包安装

1. 下载如下aclruntime和ais_bench推理程序的whl包。

   0.0.2版本（aclruntime包请根据当前环境选择适配版本）：

   |whl包|commit节点|MD5|SHA256|
   |---|---|---|---|
   |[aclruntime-0.0.2-cp37-cp37m-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp37-cp37m-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|E14ACDFBDD52E08F79456D9BC72D589C| F1523E25B714EF51E03D640570E8655A139DB8B9340C8DD6E4DA82D6122B2C01|
   |[aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp37-cp37m-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e| 9455E267118011CAC764ECECA3B13B64|4C1F7CD1CD767912B597EAF4F4BE296E914D43DE4AF80C6894399B7BF313A80F|
   |[aclruntime-0.0.2-cp38-cp38-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp38-cp38-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|CE23FEDB8BAC2917E7238B8E25F8E54D| 63C86CEE2C9F622FAB2F6A1AA4EAB47D2D68622EC12BDC8F74A9F8CED6506D67|
   |[aclruntime-0.0.2-cp38-cp38-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp38-cp38-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|52CA43514A7373E50678A890D085C531|20AFB7A24DB774EF67250E062A0F593E419DBC5A1A668B98B60D4BBF8CA87E88|
   |[aclruntime-0.0.2-cp39-cp39-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp39-cp39-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|55016F7E2544849E128AA7B5A608893D| 22824F38CAA547805FA76DBAA4889307BE171B79CCDA68AD00FED946762E6EAD|
   |[aclruntime-0.0.2-cp39-cp39-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp39-cp39-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|080065E702277C1EE443B02C902B49E6|258CDCFBBA145E200D08F1976C442BC921D68961157BDFD1F0D73985FDC45F24|
   |[aclruntime-0.0.2-cp310-cp310-linux_x86_64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp310-cp310-linux_x86_64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|78242C34E7DB95E6587C47254E309BBB|4F563603FCFF9CBC3FF74322936894C0E01038BF0101E85F03975B8BDDC57E6A|
   |[aclruntime-0.0.2-cp310-cp310-linux_aarch64.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp310-cp310-linux_aarch64.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|5988B1565C8136BF17374FA703BE0BC7|185CBC5DDE9C03E26494871FCC0A6F91351DE654CB36F9438DDBF9637C049CB8|
   |[ais_bench-0.0.2-py3-none-any.whl](https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl)|3baadae72c2afd61697fa391f0bb23807e336e9e|1E43A8BE245C015B47C9C5E72EA5F619|D52406D0AC02F9A8EBEFDCE0866736322753827298A4FCB1C23DA98789BF8EFE|


2. 执行如下命令，进行安装。

   ```bash
   # 安装aclruntime
   pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
   # 安装ais_bench推理程序
   pip3 install ./ais_bench-{version}-py3-none-any.whl
   ```

   {version}表示软件版本号，{python_version}表示Python版本号，{arch}表示CPU架构。

   说明：若为覆盖安装，请增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl --force-reinstall
   pip3 install ./ais_bench-{version}-py3-none-any.whl --force-reinstall
   ```

   分别提示如下信息则表示安装成功：

   ```bash
   # 成功安装aclruntime
   Successfully installed aclruntime-{version}
   # 成功安装ais_bench推理程序
   Successfully installed ais_bench-{version}
   ```



#### 一键式编译安装

1. **安装aclruntime包**

   在安装环境执行如下命令安装aclruntime包：

   ```bash
   pip3 install -v 'git+https://gitee.com/ascend/ait.git#egg=aclruntime&subdirectory=ait/components/benchmark/backend'
   ```

   说明：若为覆盖安装，请增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install -v --force-reinstall 'git+https://gitee.com/ascend/ait.git#egg=aclruntime&subdirectory=ait/components/benchmark/backend'
   ```

   提示如下示例信息则表示安装成功：

   ```bash
   Successfully installed aclruntime-{version}
   ```

2. **安装ais_bench推理程序包**

   在安装环境执行如下命令安装ais_bench推理程序包：

   ```bash
   pip3 install -v 'git+https://gitee.com/ascend/ait.git#egg=ais_bench&subdirectory=ait/components/benchmark'
   ```

   说明：若为覆盖安装，请增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install -v --force-reinstall 'git+https://gitee.com/ascend/ait.git#egg=ais_bench&subdirectory=ait/components/benchmark'
   ```

   提示如下示例信息则表示安装成功：

   ```bash
   Successfully installed ais_bench-{version}
   ```



#### 源代码编译安装
1. 从代码开源仓[Gitee](https://gitee.com/ascend/ait)克隆/下载工具压缩包“ait-master.zip”。

2. 将工具压缩包上传并解压至安装环境。

3. 从工具解压目录下进入/ait/ait/components/benchmark/ais_bench目录下，执行如下命令进行编译：

   ```bash
   # 进入工具解压目录
   cd ${HOME}/ait/ait/components/benchmark
   # 构建aclruntime包
   pip3 wheel ./backend/ -v
   # 构建ais_bench推理程序包
   pip3 wheel ./ -v
   ```

   其中，${HOME}为ais_bench推理工具包所在目录。

   分别提示如下信息则表示编译成功：

   ```bash
   # 成功编译aclruntime包
   Successfully built aclruntime
   # 成功编译ais_bench推理程序包
   Successfully built ais-bench
   ```

4. 执行如下命令，进行安装。

   ```bash
   # 安装aclruntime
   pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
   # 安装ais_bench推理程序
   pip3 install ./ais_bench-{version}-py3-none-any.whl
   ```

   {version}表示软件版本号，{python_version}表示Python版本号，{arch}表示CPU架构。

   说明：若为覆盖安装，请增加“--force-reinstall”参数强制安装，例如：

   ```bash
   pip3 install ./aclruntime-{version}-{python_version}-linux_{arch}.whl --force-reinstall
   pip3 install ./ais_bench-{version}-py3-none-any.whl --force-reinstall
   ```

   分别提示如下信息则表示安装成功：

   ```bash
   # 成功安装aclruntime
   Successfully installed aclruntime-{version}
   # 成功安装ais_bench推理程序
   Successfully installed ais_bench-{version}
   ```
5. 完成ais_bench推理工具安装后，需要执行如下操作，确保工具能够正确运行：
   执行requirements.txt文件中的依赖安装，执行如下命令：

   ```bash
   cd ${HOME}ait/ait/components/benchmark/
   pip3 install -r ./requirements.txt
   ```

   其中，${HOME}为ais_bench推理工具包所在目录。

   说明：若依赖已安装，忽略此步骤。


### 运行准备
1. 设置CANN包的环境变量，执行如下命令：

   ```bash
   source ${INSTALL_PATH}/Ascend/ascend-toolkit/set_env.sh
   ```

   其中，${INSTALL_PATH}为CANN包安装路径。

   说明：若环境变量已配置，忽略此步骤。

完成以上设置后，可以使用ais_bench推理工具进行推理模型的性能测试。

## 使用方法

### 工具介绍

 #### 使用入口

ais_bench推理工具可以通过ais_bench可执行文件方式启动模型测试。启动方式如下：

```bash
python3 -m ais_bench --model *.om
```
其中，*为OM离线模型文件名。

#### 参数说明

ais_bench推理工具可以通过配置不同的参数，来应对各种测试场景以及实现其他辅助功能。

参数按照功能类别分为**基础功能参数**和**高级功能参数**：

- **基础功能参数**：主要包括输入输入文件及格式、debug、推理次数、预热次数、指定运行设备以及帮助信息等。
- **高级功能参数**：主要包括动态分档场景和动态Shape场景的ais_bench推理测试参数以及profiler或dump数据获取等。

**说明**：以下参数中，参数和取值之间可以用“ ”空格分隔也可以用“=”等号分隔。例如：--debug 1或--debug=0。

##### 基础功能参数

| 参数名                | 说明                                                         | 是否必选 |
| --------------------- | ------------------------------------------------------------ | -------- |
| --model               | 需要进行推理的OM离线模型文件。                               | 是       |
| --input               | 模型需要的输入。可指定输入文件所在目录或直接指定输入文件。支持输入文件格式为“NPY”、“BIN”。可输入多个文件或目录，文件或目录之间用“,”隔开。具体输入文件请根据模型要求准备。  若不配置该参数，会自动构造输入数据，输入数据类型由--pure_data_type参数决定。 | 否       |
| --pure_data_type      | 纯推理数据类型。取值为：“zero”、“random”，默认值为"zero"。 未配置模型输入文件时，工具自动构造输入数据。设置为zero时，构造全为0的纯推理数据；设置为random时，为每一个输入生成一组随机数据。 | 否       |
| --output              | 推理结果保存目录。配置后会创建“日期+时间”的子目录，保存输出结果。如果指定output_dirname参数，输出结果将保存到子目录output_dirname下。不配置输出目录时，仅打印输出结果，不保存输出结果。 | 否       |
| --output_dirname      | 推理结果保存子目录。设置该值时输出结果将保存到*output/output_dirname*目录下。  配合output参数使用，单独使用无效。 例如：--output */output* --output_dirname *output_dirname* | 否       |
| --outfmt              | 输出数据的格式。取值为：“NPY”、“BIN”、“TXT”，默认为”BIN“。  配合output参数使用，单独使用无效。 例如：--output */output* --outfmt NPY。 | 否       |
| --debug               | 调试开关。可打印model的desc信息和其他详细执行信息。1或true（开启）、0或false（关闭），默认关闭。 | 否       |
| --run_mode | 推理执行前的数据加载方式：可取值：array（将数据转换成host侧的ndarray，再调用推理接口推理），files（将文件直接加载进device内，再调用推理接口推理），tensor（将数据加载进device内，再调用推理接口推理），full（将数据转换成host侧的ndarray，再将ndarray格式数据加载进device内，再调用推理接口推理），默认为array。 | 否 |
| --display_all_summary | 是否显示所有的汇总信息，包含h2d和d2h信息。1或true（开启）、0或false（关闭），默认关闭。 | 否       |
| --loop                | 推理次数。默认值为1，取值范围为大于0的正整数。  profiler参数配置为true时，推荐配置为1。 | 否       |
| --warmup_count        | 推理预热次数。默认值为1，取值范围为大于等于0的整数。配置为0则表示不预热。 | 否       |
| --device              | 指定运行设备。根据设备实际的Device ID指定，默认值为0。多Device场景下，可以同时指定多个Device进行推理测试，例如：--device 0,1,2,3。 | 否       |
| --divide_input | 输入数据集切分开关，1或true（开启）、0或false（关闭），默认关闭。多Device场景下，打开时，工具会将数据集平分给这些Device进行推理。| 否 |
| --help                | 工具使用帮助信息。                                           | 否       |

##### 高级功能参数

| 参数名                   | 说明                                                         | 是否必选 |
| ------------------------ | ------------------------------------------------------------ | -------- |
| --dymBatch               | 动态Batch参数，指定模型输入的实际Batch。 <br>如ATC模型转换时，设置--input_shape="data:-1,600,600,3;img_info:-1,3" --dynamic_batch_size="1,2,4,8"，dymBatch参数可设置为：--dymBatch 2。 | 否       |
| --dymHW                  | 动态分辨率参数，指定模型输入的实际H、W。 <br>如ATC模型转换时，设置--input_shape="data:8,3,-1,-1;img_info:8,4,-1,-1" --dynamic_image_size="300,500;600,800"，dymHW参数可设置为：--dymHW 300,500。 | 否       |
| --dymDims                | 动态维度参数，指定模型输入的实际Shape。 <br>如ATC模型转换时，设置 --input_shape="data:1,-1;img_info:1,-1" --dynamic_dims="224,224;600,600"，dymDims参数可设置为：--dymDims "data:1,600;img_info:1,600"。 | 否       |
| --dymShape               | 动态Shape参数，指定模型输入的实际Shape。 <br>如ATC模型转换时，设置--input_shape_range="input1:\[8\~20,3,5,-1\];input2:\[5,3\~9,10,-1\]"，dymShape参数可设置为：--dymShape "input1:8,3,5,10;input2:5,3,10,10"。<br>动态Shape场景下，获取模型的输出size通常为0（即输出数据占内存大小未知），建议设置--outputSize参数。<br/>例如：--dymShape "input1:8,3,5,10;input2:5,3,10,10" --outputSize "10000,10000" | 否       |
| --dymShape_range         | 动态Shape的阈值范围。如果设置该参数，那么将根据参数中所有的Shape列表进行依次推理，得到汇总推理信息。<br/>配置格式为：name1:1,3,200\~224,224-230;name2:1,300。其中，name为模型输入名，“\~”表示范围，“-”表示某一位的取值。<br/>也可以指定动态Shape的阈值范围配置文件*.info，该文件中记录动态Shape的阈值范围。 | 否       |
| --outputSize             | 指定模型的输出数据所占内存大小，多个输出时，需要为每个输出设置一个值，多个值之间用“,”隔开。<br>动态Shape场景下，获取模型的输出size通常为0（即输出数据占内存大小未知），需要根据输入的Shape，预估一个较合适的大小，配置输出数据占内存大小。<br>例如：--dymShape "input1:8,3,5,10;input2:5,3,10,10" --outputSize "10000,10000" | 否       |
| --auto_set_dymdims_mode  | 自动设置动态Dims模式。1或true（开启）、0或false（关闭），默认关闭。<br/>针对动态档位Dims模型，根据输入的文件的信息，自动设置Shape参数，注意输入数据只能为npy文件，因为bin文件不能读取Shape信息。<br/>配合input参数使用，单独使用无效。<br/>例如：--input 1.npy --auto_set_dymdims_mode 1 | 否       |
| --auto_set_dymshape_mode | 自动设置动态Shape模式。取值为：1或true（开启）、0或false（关闭），默认关闭。<br>针对动态Shape模型，根据输入的文件的信息，自动设置Shape参数，注意输入数据只能为npy文件，因为bin文件不能读取Shape信息。<br>配合input参数使用，单独使用无效。<br/>例如：--input 1.npy --auto_set_dymshape_mode 1 | 否       |
| --profiler               | profiler开关。1或true（开启）、0或false（关闭），默认关闭。<br>profiler数据在--output参数指定的目录下的profiler文件夹内。配合--output参数使用，单独使用无效。不能与--dump同时开启。<br/>若环境配置了AIT_NO_MSPROF_MODE=1，则使用--profiler参数采集性能数据时调用的是acl.json文件。 | 否       |
| --profiler_rename        | 调用profiler落盘文件文件名修改开关，开启后落盘的文件名包含模型名称信息。1或true（开启）、0或false（关闭），默认开启。配合--profiler参数使用，单独使用无效。 |否|
| --dump                   | dump开关。1或true（开启）、0或false（关闭），默认关闭。<br>dump数据在--output参数指定的目录下的dump文件夹内。配合--output参数使用，单独使用无效。不能与--profiler同时开启。 | 否       |
| --acl_json_path          | acl.json文件路径，须指定一个有效的json文件。该文件内可配置profiler或者dump。当配置该参数时，--dump和--profiler参数无效。 | 否       |
| --batchsize              | 模型batchsize。不输入该值将自动推导。当前推理模块根据模型输入和文件输出自动进行组Batch。参数传递的batchszie有且只用于结果吞吐率计算。自动推导逻辑为尝试获取模型的batchsize时，首先获取第一个参数的最高维作为batchsize； 如果是动态Batch的话，更新为动态Batch的值；如果是动态dims和动态Shape更新为设置的第一个参数的最高维。如果自动推导逻辑不满足要求，请务必传入准确的batchsize值，以计算出正确的吞吐率。 | 否       |
| --output_batchsize_axis  | 输出tensor的batchsize轴，默认值为0。输出结果保存文件时，根据哪个轴进行切割推理结果，比如batchsize为2，表示2个输入文件组batch进行推理，那输出结果的batch维度是在哪个轴。默认为0轴，按照0轴进行切割为2份，但是部分模型的输出batch为1轴，所以要设置该值为1。 | 否       |
| --aipp_config|带有动态aipp配置的om模型在推理前需要配置的AIPP具体参数，以.config文件路径形式传入。当om模型带有动态aipp配置时，此参数为必填参数；当om模型不带有动态aipp配置时，配置此参数不影响正常推理。|否|
| --backend|指定trtexec开关。需要指定为trtexec。配合--perf参数使用，单独使用无效。|否|
| --perf|调用trtexec开关。1或true（开启）、0或false（关闭），默认关闭。配合--backend参数使用，单独使用无效。|否|
| --energy_consumption     |能耗采集开关。1或true（开启）、0或false（关闭），默认关闭。需要配合--npu_id参数使用，默认npu_id为0。|否|
| --npu_id                 |指定npu_id，默认值为0。需要通过npu-smi info命令获取指定device所对应的npu id。配合--energy_consumption参数使用，单独使用无效。|否|
| --pipeline               |指定pipeline开关，用于开启多线程推理功能。1或true（开启）、0或false（关闭），默认关闭。|否|
| --dump_npy               |指定dump_npy开关，用于开启dump结果自动转换功能。1或true（开启）、0或false（关闭），默认关闭。需要配合--output和--dump/--acl_json_path参数使用，单独使用无效。|否|
| --threads                |指定threads开关，用于设置多计算线程推理时计算线程的数量。默认值为1，取值范围为大于0的正整数。需要配合--pipeline 1参数使用，单独使用无效。|否|

### 使用场景

 #### 纯推理场景

默认情况下，构造全为0的数据送入模型推理。

示例命令如下：

```bash
python3 -m ais_bench --model /home/model/resnet50_v1.om --output ./ --outfmt BIN --loop 5
```

#### 调试模式
开启debug调试模式。

示例命令如下：

```bash
python3 -m ais_bench --model /home/model/resnet50_v1.om --output ./ --debug 1
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

 #### 文件输入场景

使用--input参数指定模型输入文件，多个文件之间通过“,”进行分隔。

本场景会根据文件输入size和模型实际输入size进行对比，若缺少数据则会自动构造数据补全，称为组Batch。

示例命令如下：

```bash
python3 -m ais_bench --model ./resnet50_v1_bs1_fp32.om --input "./1.bin,./2.bin,./3.bin,./4.bin,./5.bin"
```

 #### 文件夹输入场景

使用input参数指定模型输入文件所在目录，多个目录之间通过“,”进行分隔。

本场景会根据文件输入size和模型实际输入size进行组Batch。

```bash
python3 -m ais_bench --model ./resnet50_v1_bs1_fp32.om --input "./"
```

模型输入需要与传入文件夹的个数一致。

例如，bert模型有三个输入，则必须传入3个文件夹，且三个文件夹分别对应模型的三个输入，顺序要对应。
模型输入参数的信息可以通过开启调试模式查看，bert模型的三个输入依次为input_ids、 input_mask、 segment_ids，所以依次传入三个文件夹：

- 第一个文件夹“./data/SQuAD1.1/input_ids"，对应模型第一个参数"input_ids"的输入
- 第二个文件夹"./data/SQuAD1.1/input_mask"，对应第二个输入"input_mask"的输入
- 第三个文件夹"./data/SQuAD1.1/segment_ids"，对应第三个输入"segment_ids"的输入

```bash
python3 -m ais_bench --model ./save/model/BERT_Base_SQuAD_BatchSize_1.om --input ./data/SQuAD1.1/input_ids,./data/SQuAD1.1/input_mask,./data/SQuAD1.1/segment_ids
```



#### 多Device场景

多Device场景下，可以同时指定多个Device进行推理测试。

示例命令如下：

```bash
python3 -m ais_bench --model ./pth_resnet50_bs1.om --input ./data/ --device 1,2
```

输出结果依次展示每个Device的推理测试结果，示例如下：

```bash
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 2.4769999980926514, max = 3.937000036239624, mean = 3.5538000106811523, median = 3.7230000495910645, percentile(99%) = 3.936680030822754
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(3.5538000106811523): 281.38893494131406
[INFO] ------------------------------------------------------
[INFO] -----------------Performance Summary------------------
[INFO] NPU_compute_time (ms): min = 3.3889999389648438, max = 3.9230000972747803, mean = 3.616000032424927, median = 3.555000066757202, percentile(99%) = 3.9134000968933105
[INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(3.616000032424927): 276.54867008654026
[INFO] ------------------------------------------------------
[INFO] unload model success, model Id is 1
[INFO] unload model success, model Id is 1
[INFO] end to destroy context
[INFO] end to destroy context
[INFO] end to reset device is 2
[INFO] end to reset device is 2
[INFO] end to finalize acl
[INFO] end to finalize acl
[INFO] multidevice run end qsize:4 result:1
i:0 device_1 throughput:281.38893494131406 start_time:1676875630.804429 end_time:1676875630.8303885
i:1 device_2 throughput:276.54867008654026 start_time:1676875630.8043878 end_time:1676875630.8326817
[INFO] summary throughput:557.9376050278543
```

其中结果最后展示每个Device推理测试的throughput（吞吐率）、start_time（测试启动时间）、end_time（测试结束时间）以及summary throughput（吞吐率汇总）。其他详细字段解释请参见本手册的“输出结果”章节。

 #### 动态分档场景

主要包含动态Batch、动态HW（宽高）、动态Dims三种场景，需要分别传入dymBatch、dymHW、dymDims指定实际档位信息。

##### 动态Batch

以档位1 2 4 8档为例，设置档位为2，本程序将获取实际模型输入组Batch，每2个输入为一组，进行组Batch。

```bash
python3 -m ais_bench --model ./resnet50_v1_dynamicbatchsize_fp32.om --input=./data/ --dymBatch 2
```

##### 动态HW宽高

以档位224,224;448,448档为例，设置档位为224,224，本程序将获取实际模型输入组Batch。

```bash
python3 -m ais_bench --model ./resnet50_v1_dynamichw_fp32.om --input=./data/ --dymHW 224,224
```

##### 动态Dims

以设置档位1,3,224,224为例，本程序将获取实际模型输入组Batch。

```bash
python3 -m ais_bench --model resnet50_v1_dynamicshape_fp32.om --input=./data/ --dymDims actual_input_1:1,3,224,224
```

##### 自动设置Dims模式（动态Dims模型）

动态Dims模型输入数据的Shape可能是不固定的，比如一个输入文件Shape为1,3,224,224，另一个输入文件Shape为 1,3,300,300。若两个文件同时推理，则需要设置两次动态Shape参数，当前不支持该操作。针对该场景，增加auto_set_dymdims_mode模式，可以根据输入文件的Shape信息，自动设置模型的Shape参数。

```bash
python3 -m ais_bench --model resnet50_v1_dynamicshape_fp32.om --input=./data/ --auto_set_dymdims_mode 1
```



#### 动态Shape场景

##### 动态Shape

以ATC设置[1\~8,3,200\~300,200\~300]，设置档位1,3,224,224为例，本程序将获取实际模型输入组Batch。

动态Shape的输出大小通常为0，建议通过outputSize参数设置对应输出的内存大小。

```bash
python3 -m ais_bench --model resnet50_v1_dynamicshape_fp32.om --dymShape actual_input_1:1,3,224,224 --outputSize 10000
```

##### 自动设置Shape模式（动态Shape模型）

动态Shape模型输入数据的Shape可能是不固定的，比如一个输入文件Shape为1,3,224,224 另一个输入文件Shape为 1,3,300,300。若两个文件同时推理，则需要设置两次动态Shape参数，当前不支持该操作。针对该场景，增加auto_set_dymshape_mode模式，可以根据输入文件的Shape信息，自动设置模型的Shape参数。

```bash
python3 -m ais_bench --model ./pth_resnet50_dymshape.om  --outputSize 100000 --auto_set_dymshape_mode 1  --input ./dymdata
```

**注意该场景下的输入文件必须为npy格式，如果是bin文件将获取不到真实的Shape信息。**

##### 动态Shape模型range测试模式

输入动态Shape的range范围。对于该范围内的Shape分别进行推理，得出各自的性能指标。

以对1,3,224,224 1,3,224,225 1,3,224,226进行分别推理为例，命令如下：

```bash
python3 -m ais_bench --model ./pth_resnet50_dymshape.om  --outputSize 100000 --dymShape_range actual_input_1:1,3,224,224~226
```

#### 动态AIPP场景
- 动态AIPP的介绍参考[ATC模型转换](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/download)中"6.1 AIPP使能"章节。
- 目前benchmark工具只支持单个input的带有动态AIPP配置的模型，只支持静态shape、动态batch、动态宽高三种场景，不支持动态shape场景。
##### --aipp_config 输入的.config文件模板
以resnet18模型所对应的一种aipp具体配置为例(actual_aipp_conf.config)：
```cfg
[aipp_op]
    input_format : RGB888_U8
    src_image_size_w : 256
    src_image_size_h : 256

    crop : 1
    load_start_pos_h : 16
    load_start_pos_w : 16
    crop_size_w : 224
    crop_size_h : 224

    padding : 0
    csc_switch : 0
    rbuv_swap_switch : 0
    ax_swap_switch : 0
    csc_switch : 0

	  min_chn_0 : 123.675
	  min_chn_1 : 116.28
	  min_chn_2 : 103.53
	  var_reci_chn_0 : 0.0171247538316637
	  var_reci_chn_1 : 0.0175070028011204
	  var_reci_chn_2 : 0.0174291938997821
```
- .config文件`[aipp_op]`下的各字段名称及其取值范围参考[ATC模型转换](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/download)中"6.1.9 配置文件模板"章节中"静态AIPP需设置，动态AIPP无需设置"部分，其中字段取值为为true、false的字段，在.config文件中取值对应为1、0。
- .config文件`[aipp_op]`下的`input_format`、`src_image_size_w`、`src_image_size_h`字段是必填字段。
- .config文件中字段的具体取值是否适配对应的模型，benchmark本身不会检测，在推理时acl接口报错不属于benchmark的问题
##### 静态shape场景示例，以resnet18模型为例
###### atc命令转换出带动态aipp配置的静态shape模型
```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_bs4_dym_aipp --input_format=NCHW --input_shape="image:4,3,224,224" --soc_version=Ascend310 --insert_op_conf=dym_aipp_conf.aippconfig --enable_small_channel=1
```
- dym_aipp_conf.aippconfig的内容(下同)为：
```
aipp_op{
    related_input_rank ： 0
    aipp_mode : dynamic
    max_src_image_size : 4000000
}
```
###### benchmark命令
```
python3 -m ais_bench --model resnet18_bs4_dym_aipp.om --aipp_config actual_aipp_conf.config
```
##### 动态batch场景示例，以resnet18模型为例
###### atc命令转换出带动态aipp配置的动态batch模型
```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_batch_aipp --input_format=NCHW --input_shape="image:-1,3,224,224" --dynamic_batch_size "1,2" --soc_version=Ascend310 --insert_op_conf=dym_aipp_conf.aippconfig --enable_small_channel=1
```
###### benchmark命令
```
python3 -m ais_bench --model resnet18_dym_batch_aipp.om --aipp_config actual_aipp_conf.config --dymBatch 1
```
##### 动态宽高场景示例，以resnet18模型为例
###### atc命令转换出带动态aipp配置的动态宽高模型
```
atc --framework=5 --model=./resnet18.onnx --output=resnet18_dym_image_aipp --input_format=NCHW --input_shape="image:4,3,-1,-1" --dynamic_image_size "112,112;224,224" --soc_version=Ascend310 --insert_op_conf=dym_aipp_conf.aippconfig --enable_small_channel=1
```
###### benchmark命令
```
python3 -m ais_bench --model resnet18_dym_image_aipp.om --aipp_config actual_aipp_conf.config --dymHW 112,112
```

#### trtexec场景

ais_bench支持onnx模型推理（集成trtexec）,trtexec为NVIDIA TensorRT自带工具。用户使用ais_bench拉起trtexec工具进行推理性能测试，测试过程中实时输出trtexec日志，打印在控制台，推理性能测试完成后，将性能数据输出在控制台。
##### 前置条件
推理性能测试环境需要配置有GPU，安装CANN、CUDA及TensorRT，并且trtexec可以通过命令行调用到，安装方式可参考[TensorRT](https://github.com/NVIDIA/TensorRT)。

示例命令如下：

```bash
python3 -m ais_bench --model pth_resnet50.onnx --backend trtexec --perf 1
```

输出结果推理测试结果，示例如下：

```bash
[INFO] [05/27/2023-12:05:31] [I] === Performance summary ===
[INFO] [05/27/2023-12:05:31] [I] Throughput: 120.699 qps
[INFO] [05/27/2023-12:05:31] [I] Latency: min = 9.11414 ms, max = 11.7442 ms, mean = 9.81005 ms, median = 9.76404 ms, percentile(90%) = 10.1075 ms, percentile(95%) = 10.1624 ms, percentile(99%) = 11.4742 ms
[INFO] [05/27/2023-12:05:31] [I] Enqueue Time: min = 0.516296 ms, max = 0.598633 ms, mean = 0.531443 ms, median = 0.5271 ms, percentile(90%) = 0.546875 ms, percentile(95%) = 0.564575 ms, percentile(99%) = 0.580566 ms
[INFO] [05/27/2023-12:05:31] [I] H2D Latency: min = 1.55066 ms, max = 1.57336 ms, mean = 1.55492 ms, median = 1.55444 ms, percentile(90%) = 1.55664 ms, percentile(95%) = 1.55835 ms, percentile(99%) = 1.56458 ms
[INFO] [05/27/2023-12:05:31] [I] GPU Compute Time: min = 7.54407 ms, max = 10.1723 ms, mean = 8.23978 ms, median = 8.19409 ms, percentile(90%) = 8.5354 ms, percentile(95%) = 8.59131 ms, percentile(99%) = 9.90002 ms
[INFO] [05/27/2023-12:05:31] [I] D2H Latency: min = 0.0130615 ms, max = 0.0170898 ms, mean = 0.015342 ms, median = 0.0153809 ms, percentile(90%) = 0.0162354 ms, percentile(95%) = 0.0163574 ms, percentile(99%) = 0.0168457 ms
[INFO] [05/27/2023-12:05:31] [I] Total Host Walltime: 3.02405 s
[INFO] [05/27/2023-12:05:31] [I] Total GPU Compute Time: 3.00752 s
```

**字段说明**

| 字段                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| Throughput            | 吞吐率。                    |
| Latency               | H2D 延迟、GPU 计算时间和 D2H 延迟的总和。这是推断单个执行的延迟。。                    |
| min                   | 推理执行时间最小值。                                         |
| max                   | 推理执行时间最大值。                                         |
| mean                  | 推理执行时间平均值。                                         |
| median                | 推理执行时间取中位数。                                       |
| percentile(99%)       | 推理执行时间中的百分位数。                                   |
| H2D Latency           | 单个执行的输入张量的主机到设备数据传输的延迟。                                   |
| GPU Compute Time      | 为执行 CUDA 内核的 GPU 延迟。                                |
| D2H Latency           | 单个执行的输出张量的设备到主机数据传输的延迟。                    |
| Total Host Walltime   | 从第一个执行（预热后）入队到最后一个执行完成的主机时间。 |
| Total GPU Compute Time| 所有执行的 GPU 计算时间的总和。 |

#### profiler或dump场景

支持以--acl_json_path、--profiler、--dump参数形式实现：
+ acl_json_path参数指定acl.json文件，可以在该文件中对应的profiler或dump参数。示例代码如下：

  + profiler

    ```bash
    {
    "profiler": {
                  "switch": "on",
                  "output": "./result/profiler"
                }
    }
    ```

    更多性能参数配置请依据CANN包种类（商用版或社区版）分别参见《[CANN 商用版：开发工具指南/性能数据采集（acl.json配置文件方式）](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0086.html)》和《[CANN 社区版：开发工具指南/性能数据采集（acl.json配置文件方式）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC1alpha002/developmenttools/devtool/atlasprofiling_16_0086.html)》中的参数配置详细描述

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

- 通过该方式进行profiler采集时，如果配置了环境变量`export AIT_NO_MSPROF_MODE=1`，输出的性能数据文件需要参见《[CANN 开发工具指南/数据解析与导出/Profiling数据导出](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0100.html)》，将性能数据解析并导出为可视化的timeline和summary文件。
- 通过该方式进行profiler采集时，如果**没有**配置环境变量`AIT_NO_MSPROF_MODE=1`，benchmark会将acl.json中与profiler相关的参数解析成msprof命令，调用msprof采集性能数据，结果默认带有可视化的timeline和summary文件，msprof输出的文件含义参考[性能数据采集（msprof命令行方式）](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasprofiling_16_0040.html)。
- 如果acl.json文件中同时配置了profiler和dump参数，需要要配置环境变量`export AIT_NO_MSPROF_MODE=1`保证同时采集

+ profiler为固化到程序中的一组性能数据采集配置，生成的性能数据保存在--output参数指定的目录下的profiler文件夹内。

  该参数是通过调用ais_bench/infer/__main__.py中的msprof_run_profiling函数来拉起msprof命令进行性能数据采集的。若需要修改性能数据采集参数，可根据实际情况修改msprof_run_profiling函数中的msprof_cmd参数。示例如下：

  ```bash
  msprof_cmd="{} --output={}/profiler --application=\"{}\" --model-execution=on --sys-hardware-mem=on --sys-cpu-profiling=off --sys-profiling=off --sys-pid-profiling=off --dvpp-profiling=on --runtime-api=on --task-time=on --aicpu=on".format(
          msprof_bin, args.output, cmd)
  ```

  该方式进行性能数据采集时，首先检查是否存在msprof命令：

  - 若命令存在，则使用该命令进行性能数据采集、解析并导出为可视化的timeline和summary文件。
  - 若命令不存在，则msprof层面会报错，benchmark层面不检查命令内容合法性。
  - 若环境配置了AIT_NO_MSPROF_MODE=1，则使用--profiler参数采集性能数据时调用的是acl.json文件。

  msprof命令不存在或环境配置了AIT_NO_MSPROF_MODE=1情况下，采集的性能数据文件未自动解析，需要参见《[CANN 开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/auxiliarydevtool_0002.html)》中的“性能分析工具>高级功能>数据解析与导出”章节，将性能数据解析并导出为可视化的timeline和summary文件。

  更多性能数据采集参数介绍请参见《[CANN 开发工具指南](https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/auxiliarydevtool/auxiliarydevtool_0002.html)》中的“性能分析工具>高级功能>性能数据采集（msprof命令行方式）”章节。

+ acl_json_path优先级高于profiler和dump，同时设置时以acl_json_path为准。

+ profiler参数和dump参数，必须要增加output参数，指示输出路径。

+ profiler和dump可以分别使用，但不能同时启用。

示例命令如下：

```bash
python3 -m ais_bench --model ./resnet50_v1_bs1_fp32.om --acl_json_path ./acl.json
python3 -m ais_bench  --model /home/model/resnet50_v1.om --output ./ --dump 1
python3 -m ais_bench  --model /home/model/resnet50_v1.om --output ./ --profiler 1
```

 #### 输出结果文件保存场景

默认情况下，ais_bench推理工具执行后不保存输出结果数据文件，配置相关参数后，可生成的结果数据如下：

| 文件/目录                                | 说明                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| {文件名}.bin、{文件名}.npy或{文件名}.txt | 模型推理输出结果文件。<br/>文件命名格式：名称_输出序号.后缀。不指定input时（纯推理），名称固定为“pure_infer_data”；指定input时，名称以第一个输入的第一个名称命名；输出的序号从0开始按输出先后顺序排列；文件名后缀由--outfmt参数控制。<br/>默认情况下，会在--output参数指定的目录下创建“日期+时间”的目录，并将结果文件保存在该目录下；当指定了--output_dirname时，结果文件将直接保存在--output_dirname参数指定的目录下。<br/>指定--output_dirname参数时，多次执行工具推理会导致结果文件因同名而覆盖。 |
| xx_summary.json                          | 工具输出模型性能结果数据。默认情况下，“xx”以“日期+时间”命名；当指定了--output_dirname时，“xx”以--output_dirname指定的目录名称命名。<br/>指定--output_dirname参数时，多次执行工具推理会导致结果文件因同名而覆盖。 |
| dump                                     | dump数据文件目录。使用--dump开启dump时，在--output参数指定的目录下创建dump目录，保存dump数据文件。 |
| profiler                                 | Profiler采集性能数据文件目录。使用--profiler开启性能数据采集时，在--output参数指定的目录下创建profiler目录，保存性能数据文件。 |

- 仅设置--output参数。示例命令及结果如下：

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --output ./result
  ```

  ```bash
  result
  |-- 2022_12_17-07_37_18
  │   `-- pure_infer_data_0.bin
  `-- 2022_12_17-07_37_18_summary.json
  ```

- 设置--input和--output参数。示例命令及结果如下：

  ```bash
  # 输入的input文件夹内容如下
  ls ./data
  196608-0.bin  196608-1.bin  196608-2.bin  196608-3.bin  196608-4.bin  196608-5.bin  196608-6.bin  196608-7.bin  196608-8.bin  196608-9.bin
  ```

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --input ./data --output ./result
  ```

  ```bash
  result/
  |-- 2023_01_03-06_35_53
  |   |-- 196608-0_0.bin
  |   |-- 196608-1_0.bin
  |   |-- 196608-2_0.bin
  |   |-- 196608-3_0.bin
  |   |-- 196608-4_0.bin
  |   |-- 196608-5_0.bin
  |   |-- 196608-6_0.bin
  |   |-- 196608-7_0.bin
  |   |-- 196608-8_0.bin
  |   `-- 196608-9_0.bin
  `-- 2023_01_03-06_35_53_summary.json
  ```

- 设置--output_dirname参数。示例命令及结果如下：

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --output ./result --output_dirname subdir
  ```

  ```bash
  result
  |-- subdir
  │   `-- pure_infer_data_0.bin
  `-- subdir_summary.json
  ```

- 设置--dump参数。示例命令及结果如下：

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --output ./result --dump 1
  ```

  ```bash
  result
  |-- 2022_12_17-07_37_18
  │   `-- pure_infer_data_0.bin
  |-- dump
  `-- 2022_12_17-07_37_18_summary.json
  ```

- 设置--profiler参数。示例命令及结果如下：

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --output ./result --profiler 1
  ```

  ```bash
  result
  |-- 2022_12_17-07_56_10
  │   `-- pure_infer_data_0.bin
  |-- profiler
  │   `-- PROF_000001_20221217075609326_GLKQJOGROQGOLIIB
  `-- 2022_12_17-07_56_10_summary.json
  ```

#### 多线程推理场景

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --pipeline 1
  ```
  在单线程推理的命令行基础上加上--pipeline 1即可开启多线程推理模式，实现计算-搬运的并行，加快端到端推理速度。

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --pipeline 1 --threads 2
  ```
  在多线程推理的命令行基础上加上--threads {$number of threads}，即可开启多计算线程推理模式，实现计算-计算的并行，提高推理吞吐量。

#### dump数据自动转换场景

  ```bash
  python3 -m ais_bench --model ./pth_resnet50_bs1.om --output ./result --dump 1 --dump_npy 1
  ```
  在dump场景上加上--dump_npy 1开启自动转换dump数据模式, 需要配合--dump或者--acl_json_path参数。

  转换后dump目录

  ```bash
  result/
  |-- 2023_01_03-06_35_53/
  |-- 2023_01_03-06_35_53_summary.json
  `-- dump/
      |--20230103063551/
      |--20230103063551_npy/
  ```


### 输出结果

ais_bench推理工具执行后，打屏输出结果示例如下：

- display_all_summary=False时，打印如下：

  ```bash
  [INFO] -----------------Performance Summary------------------
  [INFO] NPU_compute_time (ms): min = 0.6610000133514404, max = 0.6610000133514404, mean = 0.6610000133514404, median = 0.6610000133514404, percentile(99%) = 0.6610000133514404
  [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6610000133514404): 1512.8592735267011
  [INFO] ------------------------------------------------------
  ```

- display_all_summary=True时，打印如下：

  ```bash
  [INFO] -----------------Performance Summary------------------
  [INFO] H2D_latency (ms): min = 0.05700000002980232, max = 0.05700000002980232, mean = 0.05700000002980232, median = 0.05700000002980232, percentile(99%) = 0.05700000002980232
  [INFO] NPU_compute_time (ms): min = 0.6650000214576721, max = 0.6650000214576721, mean = 0.6650000214576721, median = 0.6650000214576721, percentile(99%) = 0.6650000214576721
  [INFO] D2H_latency (ms): min = 0.014999999664723873, max = 0.014999999664723873, mean = 0.014999999664723873, median = 0.014999999664723873, percentile(99%) = 0.014999999664723873
  [INFO] throughput 1000*batchsize.mean(1)/NPU_compute_time.mean(0.6650000214576721): 1503.759349974173
  ```

通过输出结果可以查看模型执行耗时、吞吐率。耗时越小、吞吐率越高，则表示该模型性能越高。

**字段说明**

| 字段                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| H2D_latency (ms)      | Host to Device的内存拷贝耗时。单位为ms。                     |
| min                   | 推理执行时间最小值。                                         |
| max                   | 推理执行时间最大值。                                         |
| mean                  | 推理执行时间平均值。                                         |
| median                | 推理执行时间取中位数。                                       |
| percentile(99%)       | 推理执行时间中的百分位数。                                   |
| NPU_compute_time (ms) | NPU推理计算的时间。单位为ms。                                |
| D2H_latency (ms)      | Device to Host的内存拷贝耗时。单位为ms。                     |
| throughput            | 吞吐率。吞吐率计算公式：1000 *batchsize/npu_compute_time.mean |
| batchsize             | 批大小。本工具不一定能准确识别当前样本的batchsize，建议通过--batchsize参数进行设置。 |

## 扩展功能

### 接口开放

开放ais_bench推理工具推理Python接口。

代码示例参考https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/test/interface_sample.py

可以通过如下示例代码完成ais_bench推理工具推理操作：

```python
def infer_simple():
  device_id = 0
  session = InferSession(device_id, model_path)

  *# create new numpy data according inputs info*
  barray = bytearray(session.get_inputs()[0].realsize)
  ndata = np.frombuffer(barray)

  outputs = session.infer([ndata])
  print("outputs:{} type:{}".format(outputs, type(outputs)))

  print("static infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))
```

动态Shape推理：

```bash
def infer_dymshape():
  device_id = 0
  session = InferSession(device_id, model_path)
  ndata = np.zeros([1,3,224,224], dtype=np.float32)

  mode = "dymshape"
  outputs = session.infer([ndata], mode, custom_sizes=100000)
  print("outputs:{} type:{}".format(outputs, type(outputs)))
  print("dymshape infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))
```

多线程推理：

使用多线程推理接口时需要注意内存的使用情况，传入的input和预计output总和内存需要小于可用内存，否则程序将会异常退出。

```python
def infer_pipeline():
  device_id = 0
  session = InferSession(device_id, model_path)

  barray = bytearray(session.get_inputs()[0].realsize)
  ndata = np.frombuffer(barray)

  outputs = session.infer([[ndata]])
  print("outputs:{} type:{}".format(outputs, type(outputs)))

  print("static infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))
```


### 推理异常保存文件功能

当出现推理异常时，会写入算子执行失败的输入输出文件到**当前目录**下。同时会打印出当前的算子执行信息。利于定位分析。示例如下：

```bash
python3 -m ais_bench --model ./test/testdata/bert/model/pth_bert_bs1.om --input ./random_in0.bin,random_in1.bin,random_in2.bin
```

```bash
[INFO] acl init success
[INFO] open device 0 success
[INFO] load model ./test/testdata/bert/model/pth_bert_bs1.om success
[INFO] create model description success
[INFO] get filesperbatch files0 size:1536 tensor0size:1536 filesperbatch:1 runcount:1
[INFO] exception_cb streamId:103 taskId:10 deviceId: 0 opName:bert/embeddings/GatherV2 inputCnt:3 outputCnt:1
[INFO] exception_cb hostaddr:0x124040800000 devaddr:0x12400ac48800 len:46881792 write to filename:exception_cb_index_0_input_0_format_2_dtype_1_shape_30522x768.bin
[INFO] exception_cb hostaddr:0x124040751000 devaddr:0x1240801f6000 len:1536 write to filename:exception_cb_index_0_input_1_format_2_dtype_3_shape_384.bin
[INFO] exception_cb hostaddr:0x124040752000 devaddr:0x12400d98e400 len:4 write to filename:exception_cb_index_0_input_2_format_2_dtype_3_shape_.bin
[INFO] exception_cb hostaddr:0x124040753000 devaddr:0x12400db20400 len:589824 write to filename:exception_cb_index_0_output_0_format_2_dtype_1_shape_384x768.bin
EZ9999: Inner Error!
EZ9999  The error from device(2), serial number is 17, there is an aicore error, core id is 0, error code = 0x800000, dump info: pc start: 0x800124080041000, current: 0x124080041100, vec error info: 0x1ff1d3ae, mte error info: 0x3022733, ifu error info: 0x7d1f3266f700, ccu error info: 0xd510fef0003608cf, cube error info: 0xfc, biu error info: 0, aic error mask: 0x65000200d000288, para base: 0x124080017040, errorStr: The DDR address of the MTE instruction is out of range.[FUNC:PrintCoreErrorInfo]

# ls exception_cb_index_0_* -lh
-rw-r--r-- 1 root root  45M Jan  7 08:17 exception_cb_index_0_input_0_format_2_dtype_1_shape_30522x768.bin
-rw-r--r-- 1 root root 1.5K Jan  7 08:17 exception_cb_index_0_input_1_format_2_dtype_3_shape_384.bin
-rw-r--r-- 1 root root    4 Jan  7 08:17 exception_cb_index_0_input_2_format_2_dtype_3_shape_.bin
-rw-r--r-- 1 root root 576K Jan  7 08:17 exception_cb_index_0_output_0_format_2_dtype_1_shape_384x768.bin
```
如果有需要将生成的异常bin文件转换为npy文件，请使用[转换脚本convert_exception_cb_bin_to_npy.py](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench/test/convert_exception_cb_bin_to_npy.py).
使用方法：python3 convert_exception_cb_bin_to_npy.py --input {bin_file_path}。支持输入bin文件或文件夹。


## FAQ
使用过程中遇到问题可以参考[FAQ](https://gitee.com/ascend/ait/wikis/benchmark_FAQ/ait%20benchmark%20%E5%AE%89%E8%A3%85%E9%97%AE%E9%A2%98FAQ)
