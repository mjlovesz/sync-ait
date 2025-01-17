# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd. All rights reserved.
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

from typing import Tuple
import unittest
import numpy as np

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_split_large_kernel import KnowledgeSplitLargeKernelConv
from testcase.helper import KnowledgeTestHelper, OptimizationConfig


def make_graph(
    name,
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    kweight_shape: Tuple[int, ...],
    kernel_pads: Tuple[int, ...],
    insert_relu_before_conv: bool = False,
    insert_relu_after_conv: bool = False,
) -> OnnxGraph:
    graph = OnnxGraph(name=name)
    graph.add_input('input', np.float32, input_shape)
    graph.add_output('output', np.float32, None)

    pre_ = 'input'
    next_ = 'relu_after_out' if insert_relu_after_conv else 'output'
    if insert_relu_before_conv:
        graph.add_node(
            name='relu_before',
            op_type='Relu',
            inputs=['input'],
            outputs=['relu_before_out'],
        )
        pre_ = 'relu_before_out'

    graph.add_initializer(
        name='weight',
        value=np.random.randn(*kweight_shape).astype(np.float32)
    )
    graph.add_initializer(
        name='bias',
        value=np.random.randn(kweight_shape[0]).astype(np.float32)
    )
    graph.add_node(
        name='conv_large',
        op_type='Conv',
        inputs=[pre_, 'weight', 'bias'],
        outputs=[next_],
        attrs={
            'kernel_shape': list(kernel_shape),
            'pads': list(kernel_pads),
            'group': 1,
        }
    )

    if insert_relu_after_conv:
        graph.add_node(
            name='relu_after',
            op_type='Relu',
            inputs=[next_],
            outputs=['output'],
        )

    graph.update_map()
    graph.infer_shape()
    return graph


class TestKnowledgeSplitLargeKernel(unittest.TestCase, KnowledgeTestHelper):
    def test_basic_split(self):
        tests = [
            # small kernel
            (1, False, (1, 3, 1024), (3, ), (128, 3), (1, 2), True, True, ),
            (1, False, (1, 3, 1344, 1344), (3, 3), (1, 3, 3, 3), (0, 1, 2, 3), True, True, ),
            (1, False, (1, 3, 133, 133, 133), (3, 3, 3), (1, 3, 3, 3, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            # 1d
            (8, True, (1, 3, 71), (65, ), (12, 3, 65), (0, 1), True, True, ),
            (1, True, (16, 3, 71), (65, ), (12, 3, 65), (0, 1), True, True, ),
            (1, True, (1, 1, 71), (65, ), (1, 1, 65), (0, 1), True, True, ),
            (1, True, (1, 1, 71), (65, ), (1, 1, 65), (0, 1), False, True, ),
            (1, True, (1, 1, 71), (65, ), (1, 1, 65), (0, 1), True, False, ),
            (1, True, (1, 1, 71), (65, ), (1, 1, 65), (0, 1), False, False, ),
            # 2d
            (8, True, (1, 1, 71, 71), (65, 65), (1, 1, 65, 65), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 3, 71, 71), (65, 65), (12, 3, 65, 65), (0, 1, 2, 3), True, True, ),
            (1, True, (2, 3, 71, 71), (65, 65), (12, 3, 65, 65), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 1, 71, 71), (65, 65), (1, 1, 65, 65), (0, 1, 2, 3), False, True, ),
            (1, True, (1, 1, 71, 71), (65, 65), (1, 1, 65, 65), (0, 1, 2, 3), True, False, ),
            (1, True, (1, 1, 71, 71), (65, 65), (1, 1, 65, 65), (0, 1, 2, 3), False, False, ),
            (1, True, (1, 1, 131, 131), (65, 65), (1, 1, 65, 65), (0, 0, 0, 0), True, True, ),
            (1, True, (1, 1, 131, 131), (65, 65), (1, 1, 65, 65), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 3, 131, 131), (65, 65), (2, 3, 65, 65), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 1, 131, 131), (65, 3), (1, 1, 65, 3), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 1, 131, 131), (3, 65), (1, 1, 3, 65), (0, 1, 2, 3), True, True, ),
            # 3d
            (1, True, (1, 1, 71, 71, 71), (65, 65, 65), (1, 1, 65, 65, 65), (0, 0, 0, 0, 0, 0), True, True, ),
            (8, True, (1, 1, 71, 71, 71), (65, 65, 65), (1, 1, 65, 65, 65), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 71, 71, 71), (3, 65, 65), (1, 1, 3, 65, 65), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 71, 71, 71), (65, 3, 65), (1, 1, 65, 3, 65), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 71, 71, 71), (65, 65, 3), (1, 1, 65, 65, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 71, 71, 71), (65, 3, 3), (1, 1, 65, 3, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 71, 71, 71), (3, 65, 3), (1, 1, 3, 65, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 71, 71, 71), (3, 3, 65), (1, 1, 3, 3, 65), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 71, 71, 71), (3, 3, 65), (1, 1, 3, 3, 65), (0, 1, 2, 3, 4, 5), False, True, ),
            (1, True, (1, 1, 71, 71, 71), (3, 3, 65), (1, 1, 3, 3, 65), (0, 1, 2, 3, 4, 5), True, False, ),
            (1, True, (1, 1, 71, 71, 71), (3, 3, 65), (1, 1, 3, 3, 65), (0, 1, 2, 3, 4, 5), False, False, ),
        ]
        for count, expect, ishape, kshape, kweight, pads, before, after in tests:
            ishape_s = 'x'.join(str(i) for i in ishape)
            kshape_s = 'x'.join(str(i) for i in kshape)
            kweight_s = 'x'.join(str(i) for i in kweight)
            pads_s = 'x'.join(str(i) for i in pads)
            name_ = f'split_kernel_in{ishape_s}_ks{kshape_s}_kw{kweight_s}_p{pads_s}_b{int(before)}_a{int(after)}'
            onnx_ori = f'./{name_}.onnx'
            graph = make_graph(name_, ishape, kshape, kweight, pads, before, after)
            for threshold in [16, 32, 48]:
                with self.subTest(name=name_):
                    onnx_opt = f'./{name_}_th{threshold}.onnx'
                    # change threshold to small number to speed up unittest
                    cfg = OptimizationConfig(
                        graph=graph,
                        knowledge=KnowledgeSplitLargeKernelConv(threshold=threshold),
                        onnx_ori=onnx_ori,
                        onnx_opt=onnx_opt,
                    )
                    self.assertTrue(self.check_optimization(cfg=cfg, expect=expect))
                    if not expect:
                        continue

                    feeds = [
                        {
                            'input': np.random.randn(*ishape).astype(np.float32)
                        }
                        for _ in range(count)
                    ]
                    self.assertTrue(
                        self.check_precision(
                            onnx_ori,
                            onnx_opt,
                            feeds,
                            cos_th=1e-3,
                            rtol=1e-2,
                            atol=1e-4
                        )
                    )


if __name__ == '__main__':
    unittest.main()
