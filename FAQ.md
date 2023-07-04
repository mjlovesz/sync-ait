### 安装ait时，出现skl2onnx组件安装失败的情况
- **解决方法**1：更换pip源，自行手动安装skl2onnx。
    命令：
    ```
    pip3 install skl2onnx==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --force-reinstall
    ```
    **解决方法**2: 直接安装wheel包[skl2onnx](https://pypi.tuna.tsinghua.edu.cn/packages/5e/59/0a47737c195da98d33f32073174b55ba4caca8b271fe85ec887463481f67/skl2onnx-1.14.1-py2.py3-none-any.whl)