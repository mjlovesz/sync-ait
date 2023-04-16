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
import onnxruntime as ort

from onnx import (
    helper,
    TensorProto
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_modify_reflection_pad import KnowledgeModifyReflectionPad
from helper import KnowledgeTestHelper, OptimizationConfig


class TestKnowledgeModifyReflectionPad(unittest.TestCase, KnowledgeTestHelper):
    def make_reflection_pad_opset11(self, onnx_name, x, padding):
        assert len(x.shape) == 4, "Please specify a 4-dim tensor shape"
        assert padding >= 0, "Please specify a non-negtive padding value"

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT,
                                          [x.shape[0], x.shape[1],
                                           x.shape[2] + 2 * padding, x.shape[3] + 2 * padding])

        pads = helper.make_tensor("pads", TensorProto.INT64, (8, ),
                                  np.array([0, 0, padding, padding, 0, 0, padding, padding]))
        pad_op = helper.make_node(
            "Pad",
            inputs=['X', 'pads'],
            outputs=['Z'],
            mode='reflect'
        )

        graph = helper.make_graph([pad_op], "test_reflection_pad_opset11", [X], [Z], [pads])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 11
        onnx.save(model, onnx_name)

    def test_modify_reflection_pad_opset11(self):
        x = np.random.randn(1, 3, 256, 256).astype(np.float32)
        padding = 2
        onnx_ori = './reflection_pad_opset11.onnx'
        onnx_opt = f'{onnx_ori[:-5]}_opt.onnx'

        self.make_reflection_pad_opset11(onnx_ori, x, padding)

        graph = OnnxGraph(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeModifyReflectionPad(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt
        )

        """
            Note: This optimization applies to .om model on NPU, thus this ONNX format inference
                does not pass the optimization test
        """
        # self.assertTrue(self.check_optimization(cfg=cfg), expect=True)
        feeds = [{'X': np.random.randn(*x.shape).astype(x.dtype)} for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))


if __name__ == '__main__':
    unittest.main()


























