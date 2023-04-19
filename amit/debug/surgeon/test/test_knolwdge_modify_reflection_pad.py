# Copyright 2022 Huawei Technologies Co. Ltd.
#
# Licensed under Apache Licenses, version 2.0 (the "License")
# you may not use the file except in compliance with the License
# you may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import onnx

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_modify_reflection_pad import KnowledgeModifyReflectionPad

from auto_optimizer.graph_optimizer import GraphOptimizer
from helper import KnowledgeTestHelper, OptimizationConfig


def make_reflection_pad_model(model_name):
    padding = 2
    x = np.random.randn(1, 3, 256, 256).astype(np.float32)
    input_x = onnx.helper.make_tensor_value_info("input_x", onnx.TensorProto.FLOAT, x.shape)
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT,
                                                [x.shape[0], x.shape[1],
                                                 x.shape[2] + 2 * padding, x.shape[3] + 2 * padding])

    pads = onnx.helper.make_tensor("pads", onnx.TensorProto.INT64, (8,),
                                   np.array([0, 0, padding, padding, 0, 0, padding, padding]))
    pad_op = onnx.helper.make_node(
        "Pad",
        inputs=['input_x', 'pads'],
        outputs=['output'],
        mode='reflect'
    )

    graph = onnx.helper.make_graph([pad_op], model_name, [input_x], [output], [pads])
    model = onnx.helper.make_model(graph)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, model_name + ".onnx")


class TestKnowledgeModifyReflectionPad(unittest.TestCase, KnowledgeTestHelper):

    def test_modify_reflection_pad_opset11(self):
        x = np.random.randn(1, 3, 256, 256).astype(np.float32)

        model_name = "test_reflection_pad"
        ori_model_path = model_name + ".onnx"
        opt_model_path = model_name + "_opt.onnx"

        make_reflection_pad_model(model_name)

        graph = OnnxGraph.parse(ori_model_path)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeModifyReflectionPad(),
            onnx_ori=ori_model_path,
            onnx_opt=opt_model_path
        )

        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{'input_x': np.random.randn(*x.shape).astype(x.dtype)}]
        self.assertTrue(self.check_precision(ori_model_path, opt_model_path, feeds))


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('10.174.180.167', port=9993, stdoutToServer=True, stderrToServer=True)
if __name__ == '__main__':
    unittest.main()
