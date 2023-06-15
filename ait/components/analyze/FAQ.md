# FAQ

## 1. 非root用户使用analyze工具时若使用root目录下/usr/local/Ascend/ascend-toolkit的文件，产生调用fast_query shell失败的错误

- 错误信息：

 

- 错误原因分析：

    当前analyze工具在检查模型支持度分析过程中会调用CANN包下算子速查工具进行检验，由于工具文件安全性检查要求调用算子速查工具脚本的使用者与该脚本的拥有者为同一人，故当非root用户使用root目录下/usr/local/Ascend/ascend-toolkit下文件时，将无法通过analyze工具的文件安全校验，所以无法调用。

- 解决方案：

    非root用户在/home/userxxx/目录下自行安装CANN开发者套件包，并正确配置相关环境变量（安装CANN包完成后根据提示），随后运行analyze工具即可。
