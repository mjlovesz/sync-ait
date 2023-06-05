# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import stat
import pytest

import numpy as np

from msquickcmp.caffe_model.caffe_dump_data import CaffeDumpData

try:
    import caffe
except ModuleNotFoundError as ee:
    print(ee)
    caffe = None

 
OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR


CAFFE_PROTOTXT = """name: "caffe_ResNet-50"\nlayer {\n name: "Input_1"\n type: "Input"\n top: "data"
  input_param {\n shape {\n dim: 1\n dim: 3\n dim: 32\n dim: 32\n}\n}\n}\n
layer {\n bottom: "data"\n top: "conv1"\n name: "conv1"\n type: "Convolution"\n convolution_param {
  num_output: 64\n kernel_size: 7\n pad: 3\n stride: 2\n }\n}\n
layer {\n bottom: "conv1"\n top: "conv1"\n name: "bn_conv1"\n type: "BatchNorm"\n batch_norm_param {
  use_global_stats: true\n }\n}\n
layer {\n bottom: "conv1"\n top: "conv1"\n name: "conv1_relu"\n type: "ReLU"\n}\n
layer {\n bottom: "conv1"\n top: "pool5"\n name: "pool5"\n type: "Pooling"\n pooling_param {
  kernel_size: 16\n stride: 1\n pool: AVE\n}\n}\n
layer {\n bottom: "pool5"\n top: "fc1000"\n name: "fc1000"\n type: "InnerProduct"\n inner_product_param {
  num_output: 1000\n    }\n}\n"""


class Args:
    def __init__(self, **kwargs):
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

@pytest.mark.skipif(caffe is None, reason="Caffe not found")
@pytest.fixture(scope="module", autouse=True)
def fake_caffe_model_args():
    import caffe
    import acl
    import shutil

    model_name = "fake_caffe_model"
    model_path = os.path.join("/tmp", model_name + ".prototxt")
    weight_path = os.path.join("/tmp", model_name + ".caffemodel")
    out_path = "/tmp/fake_caffe_model_dump"

    with os.fdopen(os.open(model_path, OPEN_FLAGS, OPEN_MODES), "w") as model_file:
        model_file.write(CAFFE_PROTOTXT)
    net = caffe.Net(model_path, caffe.TEST)
    net.save(weight_path)

    yield Args(model_path=model_path, weight_path=weight_path, out_path=out_path, input_path="", input_shape="")

    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(weight_path):
        os.remove(weight_path)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

@pytest.mark.skipif(caffe is None, reason="Caffe not found")
def test_caffe_dump_data_given_valid_when_any_then_pass(fake_caffe_model_args):
    caffe_dump = CaffeDumpData(fake_caffe_model_args)
    dump_data_dir = caffe_dump.generate_dump_data()

    assert os.path.exists(dump_data_dir)

    expect_output = ["Input_1", "bn_conv1", "conv1", "conv1_relu", "fc1000", "pool5"]
    actual_output = sorted([ii.split(".")[0] for ii in os.listdir(dump_data_dir)])
    assert actual_output == expect_output

    output_info = caffe_dump.get_net_output_info()
    assert list(output_info.keys()) == [0]
    assert output_info[0].split(".")[0] == os.path.join(dump_data_dir, "fc1000")
