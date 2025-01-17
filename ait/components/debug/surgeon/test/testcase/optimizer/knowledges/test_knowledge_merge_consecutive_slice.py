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
import onnx
from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_slice import KnowledgeMergeConsecutiveSlice
from testcase.helper import KnowledgeTestHelper, OptimizationConfig


def make_c2_slice_model(onnx_name, x):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

    start0 = helper.make_tensor("start0", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    end0 = helper.make_tensor("end0", TensorProto.INT64, [1], np.array([2], dtype=np.int64))
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    step0 = helper.make_tensor("step0", TensorProto.INT64, [1], np.array([1], dtype=np.int64))

    start1 = helper.make_tensor("start1", TensorProto.INT64, [1], np.array([1], dtype=np.int64))
    end1 = helper.make_tensor("end1", TensorProto.INT64, [1], np.array([5], dtype=np.int64))
    axes1 = helper.make_tensor("axes1", TensorProto.INT64, [1], np.array([1], dtype=np.int64))
    step1 = helper.make_tensor("step1", TensorProto.INT64, [1], np.array([1], dtype=np.int64))

    node_slice0 = helper.make_node("Slice", ["X", "start0", "end0", "axes0", "step0"], ["X_S"], "Slice0")
    node_slice1 = helper.make_node("Slice", ["X_S", "start1", "end1", "axes1", "step1"], ["Z"], "Slice1")

    graph = helper.make_graph([node_slice0, node_slice1], "continue_slice_test",
                              [input_x], [input_z], [start0, end0, axes0, step0, start1, end1, axes1, step1],)
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


def make_c2_slice_optional_args_model(onnx_name, x):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

    start0 = helper.make_tensor("start0", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    end0 = helper.make_tensor("end0", TensorProto.INT64, [1], np.array([2], dtype=np.int64))

    start1 = helper.make_tensor("start1", TensorProto.INT64, [1], np.array([1], dtype=np.int64))
    end1 = helper.make_tensor("end1", TensorProto.INT64, [1], np.array([5], dtype=np.int64))

    node_slice0 = helper.make_node("Slice", ["X", "start0", "end0"], ["X_S"], "Slice0")
    node_slice1 = helper.make_node("Slice", ["X_S", "start1", "end1"], ["Z"], "Slice1")

    graph = helper.make_graph([node_slice0, node_slice1], "continue_slice_test",
                              [input_x], [input_z], [start0, end0, start1, end1],)
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


def make_c2_slice_2dim_1dims_model(onnx_name, x, same_axis=False):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

    start0 = helper.make_tensor("start0", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    end0 = helper.make_tensor("end0", TensorProto.INT64, [1], np.array([2], dtype=np.int64))
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    step0 = helper.make_tensor("step0", TensorProto.INT64, [1], np.array([1], dtype=np.int64))

    if same_axis:
        axis = 0
    else:
        axis = 1
    start1 = helper.make_tensor("start1", TensorProto.INT64, [2], np.array([1, 3], dtype=np.int64))
    end1 = helper.make_tensor("end1", TensorProto.INT64, [2], np.array([5, 4], dtype=np.int64))
    axes1 = helper.make_tensor("axes1", TensorProto.INT64, [2], np.array([2, axis], dtype=np.int64))
    step1 = helper.make_tensor("step1", TensorProto.INT64, [2], np.array([1, 1], dtype=np.int64))

    node_slice0 = helper.make_node("Slice", ["X", "start0", "end0", "axes0", "step0"], ["X_S"], "Slice0")
    node_slice1 = helper.make_node("Slice", ["X_S", "start1", "end1", "axes1", "step1"], ["Z"], "Slice1")

    graph = helper.make_graph([node_slice0, node_slice1], "continue_slice_test",
                              [input_x], [input_z], [start0, end0, axes0, step0, start1, end1, axes1, step1],)
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


def make_c2_slice_2dim_model(onnx_name, x, same_axis=False):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

    start0 = helper.make_tensor("start0", TensorProto.INT64, [2], np.array([0, 1], dtype=np.int64))
    end0 = helper.make_tensor("end0", TensorProto.INT64, [2], np.array([2, 4], dtype=np.int64))
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [2], np.array([0, 1], dtype=np.int64))
    step0 = helper.make_tensor("step0", TensorProto.INT64, [2], np.array([1, 1], dtype=np.int64))

    if same_axis:
        axis = 1
    else:
        axis = 3
    start1 = helper.make_tensor("start1", TensorProto.INT64, [2], np.array([1, 3], dtype=np.int64))
    end1 = helper.make_tensor("end1", TensorProto.INT64, [2], np.array([5, 4], dtype=np.int64))
    axes1 = helper.make_tensor("axes1", TensorProto.INT64, [2], np.array([2, axis], dtype=np.int64))
    step1 = helper.make_tensor("step1", TensorProto.INT64, [2], np.array([1, 1], dtype=np.int64))

    node_slice0 = helper.make_node("Slice", ["X", "start0", "end0", "axes0", "step0"], ["X_S"], "Slice0")
    node_slice1 = helper.make_node("Slice", ["X_S", "start1", "end1", "axes1", "step1"], ["Z"], "Slice1")

    graph = helper.make_graph([node_slice0, node_slice1], "continue_slice_test",
                              [input_x], [input_z], [start0, end0, axes0, step0, start1, end1, axes1, step1],)
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


def make_c3_slice_model(onnx_name, x):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

    start0 = helper.make_tensor("start0", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    end0 = helper.make_tensor("end0", TensorProto.INT64, [1], np.array([2], dtype=np.int64))
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    step0 = helper.make_tensor("step0", TensorProto.INT64, [1], np.array([1], dtype=np.int64))

    start1 = helper.make_tensor("start1", TensorProto.INT64, [1], np.array([1], dtype=np.int64))
    end1 = helper.make_tensor("end1", TensorProto.INT64, [1], np.array([5], dtype=np.int64))
    axes1 = helper.make_tensor("axes1", TensorProto.INT64, [1], np.array([1], dtype=np.int64))
    step1 = helper.make_tensor("step1", TensorProto.INT64, [1], np.array([1], dtype=np.int64))

    start2 = helper.make_tensor("start2", TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    end2 = helper.make_tensor("end2", TensorProto.INT64, [1], np.array([3], dtype=np.int64))
    axes2 = helper.make_tensor("axes2", TensorProto.INT64, [1], np.array([2], dtype=np.int64))
    step2 = helper.make_tensor("step2", TensorProto.INT64, [1], np.array([1], dtype=np.int64))

    node_slice0 = helper.make_node("Slice", ["X", "start0", "end0", "axes0", "step0"], ["X_S"], "Slice0")
    node_slice1 = helper.make_node("Slice", ["X_S", "start1", "end1", "axes1", "step1"], ["X_S_S"], "Slice1")
    node_slice2 = helper.make_node("Slice", ["X_S_S", "start2", "end2", "axes2", "step2"], ["Z"], "Slice2")

    graph = helper.make_graph(
        nodes=[node_slice0, node_slice1, node_slice2],
        name="continue3_slice_test",
        inputs=[input_x],
        outputs=[input_z],
        initializer=[start0, end0, axes0, step0, start1, end1, axes1, step1, start2, end2, axes2, step2],
    )
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


def make_c4_slice_model(onnx_name, x, same_axis=False):
    input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

    inits = []
    starts = [0, 1, 0, 5]
    ends = [2, 5, 3, -1]

    for i in range(4):
        axis = 0 if same_axis else i
        inits.append(helper.make_tensor(f"start{i}", TensorProto.INT64, [1], np.array([starts[i]], dtype=np.int64)))
        inits.append(helper.make_tensor(f"end{i}", TensorProto.INT64, [1], np.array([ends[i]], dtype=np.int64)))
        inits.append(helper.make_tensor(f"axes{i}", TensorProto.INT64, [1], np.array([axis], dtype=np.int64)))
        inits.append(helper.make_tensor(f"step{i}", TensorProto.INT64, [1], np.array([1], dtype=np.int64)))

    node_slice0 = helper.make_node("Slice", ["X", "start0", "end0", "axes0", "step0"], ["X_S"], "Slice0")
    node_slice1 = helper.make_node("Slice", ["X_S", "start1", "end1", "axes1", "step1"], ["X_S_S"], "Slice1")
    node_slice2 = helper.make_node("Slice", ["X_S_S", "start2", "end2", "axes2", "step2"], ["X_S_S_S"], "Slice2")
    node_slice3 = helper.make_node("Slice", ["X_S_S_S", "start3", "end3", "axes3", "step3"], ["Z"], "Slice3")

    graph = helper.make_graph([node_slice0, node_slice1, node_slice2, node_slice3],
                              "continue4_slice_test", [input_x], [input_z], inits)
    model = helper.make_model(graph, ir_version=8)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 11
    onnx.save(model, onnx_name)


class TestKnowledgeMergeConsecutiveSlice(unittest.TestCase, KnowledgeTestHelper):

    def test_merge_c2_slice(self):
        x = np.random.randn(50, 50, 50).astype(np.float32)

        onnx_name = "c2_slice"
        onnx_ori = f"./{onnx_name}.onnx"
        onnx_opt = f"./{onnx_name}_optimize.onnx"

        make_c2_slice_model(onnx_ori, x)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{
            'X': np.random.randn(*x.shape).astype(x.dtype)
        } for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_merge_c3_slice(self):
        x = np.random.randn(50, 50, 50).astype(np.float32)

        onnx_name = "c3_slice"
        onnx_ori = f"./{onnx_name}.onnx"
        onnx_opt = f"./{onnx_name}_optimize.onnx"

        make_c3_slice_model(onnx_ori, x)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{
            'X': np.random.randn(*x.shape).astype(x.dtype)
        } for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_merge_c4_slice(self):
        x = np.random.randn(50, 50, 50, 50).astype(np.float32)

        onnx_name = "c4_slice"
        onnx_ori = f"./{onnx_name}.onnx"
        onnx_opt = f"./{onnx_name}_optimize.onnx"

        make_c4_slice_model(onnx_ori, x, False)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{
            'X': np.random.randn(*x.shape).astype(x.dtype)
        } for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_merge_c4_slice_same_axis(self):
        x = np.random.randn(50, 50, 50, 50).astype(np.float32)

        onnx_name = "c4_slice_same_axis"
        onnx_ori = f"./{onnx_name}.onnx"

        make_c4_slice_model(onnx_ori, x, True)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=False))

    def test_merge_c2_slice_2dims(self):
        x = np.random.randn(50, 50, 50, 30).astype(np.float32)

        onnx_name = "c2_slice_2dims"
        onnx_ori = f"./{onnx_name}.onnx"
        onnx_opt = f"./{onnx_name}_optimize.onnx"

        make_c2_slice_2dim_model(onnx_ori, x, False)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{
            'X': np.random.randn(*x.shape).astype(x.dtype)
        } for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_merge_c2_slice_2dims_same_axis(self):
        x = np.random.randn(50, 50, 50, 30).astype(np.float32)

        onnx_name = "c2_slice_2dims_same_axis"
        onnx_ori = f"./{onnx_name}.onnx"

        make_c2_slice_2dim_model(onnx_ori, x, True)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=False))

    def test_merge_c2_slice_2dims_1dims(self):
        x = np.random.randn(50, 50, 50, 30).astype(np.float32)

        onnx_name = "c2_slice_2dims_1dims"
        onnx_ori = f"./{onnx_name}.onnx"
        onnx_opt = f"./{onnx_name}_optimize.onnx"

        make_c2_slice_2dim_model(onnx_ori, x, False)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{
            'X': np.random.randn(*x.shape).astype(x.dtype)
        } for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_merge_c2_slice_2dims_1dims_same_axis(self):
        x = np.random.randn(50, 50, 50, 30).astype(np.float32)

        onnx_name = "c2_slice_2dims_1dims_same"
        onnx_ori = f"./{onnx_name}.onnx"

        make_c2_slice_2dim_model(onnx_ori, x, True)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=False))

    def test_merge_c2_optional_args_slice(self):
        x = np.random.randn(50, 50, 50).astype(np.float32)

        onnx_name = "c2_slice_optional_args"
        onnx_ori = f"./{onnx_name}.onnx"

        make_c2_slice_optional_args_model(onnx_ori, x)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeMergeConsecutiveSlice(),
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=False))


if __name__ == "__main__":
    unittest.main()
