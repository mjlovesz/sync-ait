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


import unittest
import numpy as np

from auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder, OnnxInitializer, OnnxNode
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from testcase.refactor.test_node_common import is_ph_equal, is_ini_equal, is_node_equal
from testcase.refactor.test_graph_basic import is_graph_equal


def create_graph(name: str = 'test_graph'):
    input_ = OnnxPlaceHolder('input', np.dtype('float32'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('output', np.dtype('float32'), [1, 3, 224, 224])
    node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['input'], outputs=['sqrt0_output'], attrs={})
    node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    node_4 = OnnxNode('flatten4', 'Flatten', inputs=['relu3_output'], outputs=['output'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[node_0, node_1, node_2, node_3, node_4],
        inputs=[input_],
        outputs=[output],
    )


def create_combined_graph(name: str = 'test_combined_graph'):
    input_ = OnnxPlaceHolder('pre_input', np.dtype('float32'), [1, 3, 224, 224])
    output = OnnxPlaceHolder('output', np.dtype('float32'), [1, 3, 224, 224])
    g1_node_0 = OnnxNode('pre_sqrt0', 'Sqrt', inputs=['pre_input'], outputs=['pre_sqrt0_output'], attrs={})
    g1_node_1 = OnnxNode('pre_relu1', 'Relu', inputs=['pre_sqrt0_output'], outputs=['pre_relu1_output'], attrs={})
    g1_node_2 = OnnxNode('pre_sqrt2', 'Sqrt', inputs=['pre_relu1_output'], outputs=['pre_sqrt2_output'], attrs={})
    g1_node_3 = OnnxNode('pre_relu3', 'Relu', inputs=['pre_sqrt2_output'], outputs=['pre_relu3_output'], attrs={})
    g1_node_4 = OnnxNode('pre_flatten4', 'Flatten', inputs=['pre_relu3_output'], outputs=['pre_output'], attrs={})

    g2_node_0 = OnnxNode('sqrt0', 'Sqrt', inputs=['pre_output'], outputs=['sqrt0_output'], attrs={})
    g2_node_1 = OnnxNode('relu1', 'Relu', inputs=['sqrt0_output'], outputs=['relu1_output'], attrs={})
    g2_node_2 = OnnxNode('sqrt2', 'Sqrt', inputs=['relu1_output'], outputs=['sqrt2_output'], attrs={})
    g2_node_3 = OnnxNode('relu3', 'Relu', inputs=['sqrt2_output'], outputs=['relu3_output'], attrs={})
    g2_node_4 = OnnxNode('flatten4', 'Flatten', inputs=['relu3_output'], outputs=['output'], attrs={})

    return OnnxGraph(
        name=name,
        nodes=[
            g1_node_0, g1_node_1, g1_node_2, g1_node_3, g1_node_4,
            g2_node_0, g2_node_1, g2_node_2, g2_node_3, g2_node_4
        ],
        inputs=[input_],
        outputs=[output],
        value_infos=[

        ]
    )


class TestGraphExtract(unittest.TestCase):
    def setUp(self):
        self.addTypeEqualityFunc(OnnxNode, is_node_equal)
        self.addTypeEqualityFunc(OnnxPlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(OnnxInitializer, is_ini_equal)
        self.addTypeEqualityFunc(OnnxGraph, is_graph_equal)
        self.graph = create_graph()
        self.combined_graph = create_combined_graph()

    def test_concatenate_graph_with_single_in_and_out(self):
        combined_graph = OnnxGraph.concat_graph(
            self.graph, self.graph,
            [("output", "input")]
        )
        print(self.combined_graph.next_map)
        self.assertEqual(combined_graph, self.combined_graph)


if __name__ == '__main__':
    unittest.main()
