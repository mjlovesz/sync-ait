## Nvjpeg 图片编码迁移
- 将使用 nvjpeg 的图片编码样例迁移到昇腾平台使用 ACL 接口实现，将一帧 yuv 格式的视频流输入编码为 jpg 格式并保存到文件。

## 准备测试图片
- 昇腾平台视频流使用的 YUV420 格式编码方式为 semi planar，即 YUV420SP，nvjpeg 使用的是 planar，即 YUV420，准备同一张图片的两种编码格式数据
- 使用 `scikit-image` 中的测试图片，也可使用自行获取的图片
  ```py
  import cv2
  import numpy as np
  from skimage.data import chelsea

  image = cv2.resize(chelsea()[:300, :400], (640, 480))
  yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)
  yuv_data = yuv_image.flatten().astype('uint8')
  yuv_data.tofile('sample_480_640_planar.yuv')

  cc = np.stack([yuv_data[640*480:640*480 + 640*480//4], yuv_data[640*480 + 640*480//4:]]).T.flatten()
  semi_yuv_data = np.concatenate([yuv_data[:640*480], cc.flatten()])
  semi_yuv_data.tofile('sample_480_640_semiplanar.yuv')
  ```
## AIT transplt 迁移分析
- 安装 ait 工具后，针对待迁移项目执行 transplt 迁移分析
```sh
ait transplt -s .
# INFO - scan_api.py[123] - Scan source files...
# ...
# INFO - csv_report.py[46] - Report generated at: ./output.xlsx
# INFO - scan_api.py[113] - **** Project analysis finished <<<
```
最终分析结果文件位于当前执行路径下 `./output.xlsx`，该结果中重点关注有对应关系的接口，并参照 `AscendAPILink` 中相关接口说明辅助完成迁移
