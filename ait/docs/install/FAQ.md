# faq
## 1、Q： 安装失败，提示“find no cann path”，如何处理？

安装报错：

![输入图片说明](https://foruda.gitee.com/images/1686801650121824710/b64bf91e_9570626.png "屏幕截图")

**A：** 安装后用户可通过 设置CANN_PATH环境变量 ，指定安装的CANN版本路径，例如：export CANN_PATH=/xxx/Ascend/ascend-toolkit/latest/。若不设置，工具默认会从环境变量ASCEND_TOOLKIT_HOME和/usr/local/Ascend/ascend-toolkit/latest路径分别尝试获取CANN版本。

    > 以下是设置CANN包环境变量的通用方法(假设CANN包安装目录为`ACTUAL_CANN_PATH`)：
    > * 执行如下命令：
    ```
    source $ACTUAL_CANN_PATH/Ascend/ascend-toolkit/set_env.sh- [目录](#目录)


## 2、Q：使用./install.sh进行安装却报-bash: ./install.sh: Permission denied
**A：** 这是因为没有给install.sh添加执行权限导致的。

```
chmod u+x install.sh
```


## 3、Q：常见报错 XXX requires YYY, which is not installed。
![which is not installed](https://foruda.gitee.com/images/1686645293870003179/234cf67c_8913618.png "屏幕截图")
**A：** 这是由于本地安装包缺乏依赖导致的，并非ait报错，根据命令行提示安装即可。

```
pip3 install YYY
```

## 4、Q：使用./install.sh，报错：/usr/bin/env: ‘bash\r’: No such file or directory。 

![No such file or directory](https://foruda.gitee.com/images/1686645345634951894/08f7e806_8913618.png "屏幕截图")

**A：** 这并不是文件报错，常见原因是因为代码在本地编译器中被默认更换了格式，在pycharm编辑器右下角将.sh文件格式由CRLF改为LF。
![CRLF改为LF](https://foruda.gitee.com/images/1686645370968699210/f44f04b3_8913618.png "屏幕截图")


## 5、Q：如何获取`cann包路径`？
**A：** 在这个命令中，export | grep ASCEND_HOME_PATH会将所有环境变量输出，并通过管道符将结果传递给grep命令。grep命令会查找包含ASCEND_HOME_PATH的行，并将结果传递给cut命令。cut命令会以等号为分隔符，提取第二个字段，即ASCEND_HOME_PATH的值，并将其输出。

```
export | grep ASCEND_HOME_PATH | cut -d'=' -f2
```

## 6、Q: 之前安装ait能够使用，后续环境上的依赖包被其他人或者其他工具破坏了，使用ait时提示“pkg_resources.VersionConflict:XXXXX”怎么办？

![输入图片说明](https://foruda.gitee.com/images/1686886830863530517/53f5816a_9570626.png "屏幕截图")

**A:** 说明ait的依赖包版本可能被升级到了不匹配版本，只需要重新安装下ait即可，即重新在ait/ait目录中，执行
```
./install.sh
```

或者执行
```
pip3 check
```
查看环境上的python组件存在哪些版本依赖不匹配，手动安装到对应版本即可，比如如下check结果表示protobuf版本不匹配，重新安装对应版本即可：

![输入图片说明](https://foruda.gitee.com/images/1686887221107606902/a0872e5b_9570626.png "屏幕截图")

执行
```
pip3 install protobuf==3.20.2
```

## 7、Q：安装ait时，出现skl2onnx组件安装失败的情况
![输入图片说明](https://foruda.gitee.com/images/1688461726292472393/721044b8_8277365.png "屏幕截图")
**A:** 
解决方法1：更换pip源，自行手动安装skl2onnx。执行
    ```
    pip3 install skl2onnx==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/  --trusted-host pypi.tuna.tsinghua.edu.cn
    ```

解决方法2：直接安装wheel包

下载[skl2onnx](https://pypi.tuna.tsinghua.edu.cn/packages/5e/59/0a47737c195da98d33f32073174b55ba4caca8b271fe85ec887463481f67/skl2onnx-1.14.1-py2.py3-none-any.whl)后，在下载好的目录中，执行
    ```
    pip3 install skl2onnx-1.14.1-py2.py3-none-any.whl
    ```
- 这里是列表文本

## 8、Q：OpenSSL: error:1408F10B:SSL routines:ssl3_get_record:wrong version number
**A:** 
解决方案：此问题为网络问题且多存在于黄区，一般配置代理为私人代理后重新安装ait即可（如果仍然不能解决并不影响ait的使用，仅影响transplt组件），代理格式如下：
    
```
    export http_proxy="http://用户名:密码@proxy.huawei.com:8080/"
    export https_proxy="http://用户名:密码@proxy.huawei.com:8080/" 
```

注：密码要用url转移