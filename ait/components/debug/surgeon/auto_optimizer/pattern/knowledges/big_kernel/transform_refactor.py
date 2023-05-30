from collections import deque
import copy

import numpy as np

from auto_optimizer.graph_refactor import Node
from auto_optimizer.pattern.knowledges.big_kernel.attention_parser import AttentionParser
from auto_optimizer.tools.log import logger
from auto_optimizer import OnnxNode, OnnxInitializer, OnnxGraph
from auto_optimizer.pattern.knowledges.big_kernel.util import (
    QK_MASK_ADD,
    CONVERT_3DIMS_TO_4DIMS,
    START_ADD,
    END_ADD,
    QK_MATMUL,
    MUL,
    MUL_B,
    QK_MASK_ADD_B,
    SOFTMAX,
    SCORE_V_MATMUL,
    TRANSPOSE,
    RESHAPE,
    RESHAPE_S,
    MATMUL,
    MATMUL_W,
    ADD,
    ADD_B,
    END_ADD_B
)


class TransformRefactor:
    def __init__(self, graph: OnnxGraph):
        self.graph = graph

    def update_graph_map(self):
        self.graph.update_map()

    def toposort_graph(self):
        self.graph.toposort()

    def infer_graph_shape(self):
        self.graph.infer_shape()

    def get_anchor_nodes(self, match_nodes, last_node):
        """
        选取几个节点作为锚点
        """
        softmax = None
        atten_start_node = None
        atten_end_node = None
        for node_name, node in match_nodes.items():
            if not atten_start_node:  # 匹配到的第一个节点就是attention的起始节点
                atten_start_node = node
                continue

            if node.op_type == "Softmax":
                softmax = node
                continue

            if node.op_type != last_node.op_type:
                continue

            is_next_node_in_subgraph = False
            for output in node.outputs:
                next_nodes = self.graph.get_next_nodes(output)
                for next_node in next_nodes:
                    if next_node.name in match_nodes:
                        is_next_node_in_subgraph = True
                        continue

            # 所有的next_node都不在子图中，则说明该节点是attention的最后一个节点
            if not is_next_node_in_subgraph:
                atten_end_node = node
        return atten_start_node, atten_end_node, softmax

    def remove_unused_attention_node(self, node: Node):
        """
        对于一个节点，bfs向下删除后面的节点以及output
        """
        queue = deque([node])
        visited = []
        while queue:
            node = queue.popleft()
            if node in self.graph.nodes:
                self.graph.nodes.remove(node)

            if node in visited:
                continue
            visited.append(node)

            for output in node.outputs:
                next_nodes = self.graph.get_next_nodes(output)
                if next_nodes:
                    queue.append(*next_nodes)
                else:
                    self.remove_output(output)

    def remove_output(self, output_name):
        for out in self.graph.outputs:
            if out.name == output_name:
                self.graph.outputs.remove(out)
                break

    def update_graph(self, attention_parser: AttentionParser, normal_subgraph: OnnxGraph):
        """
        更新原始模型的node和initializer
        """
        for node in attention_parser.graph.nodes:
            if node != attention_parser.end_node:
                next_nodes = self.graph.get_next_nodes(node.outputs[0])
                for next_node in next_nodes:
                    if next_node in attention_parser.graph.nodes:
                        continue
                    # 删除不在attention中的next_node
                    # 有些模型如gpt2，将attention的中间结果作为output，需要删除这些中间节点和output，否则匹配不上标准pattern。
                    self.remove_unused_attention_node(next_node)

            if node in self.graph.nodes and node not in normal_subgraph.nodes:
                self.graph.nodes.remove(node)

        # 将标准子图里的node和initializer加到graph中
        for init in normal_subgraph.initializers:
            self.graph.initializers.append(init)

        for node in normal_subgraph.nodes:
            self.graph.nodes.append(node)

    def insert_reshape_node(self, ref_name, reshape_name, shape, mode="after"):
        insert_reshape_s = self.graph.add_initializer(name=reshape_name+"_s",
                                                      value=np.array(shape))
        insert_reshape = self.graph.add_node(name=reshape_name, op_type="Reshape",
                                             inputs=[reshape_name + "_input", insert_reshape_s.name],
                                             outputs=[reshape_name + "_output"])

        self.graph.insert_node(refer_name=ref_name, insert_node=insert_reshape, mode=mode)
        insert_reshape.inputs.append(insert_reshape_s.name)

    def update_normal_subgraph(self, normal_subgraph: OnnxGraph, attention_parser: AttentionParser, prefix: str):
        """
        用获取到的attention子图中的参数更新标准子图中的参数
        对于一些可能存在的分支节点，如qk_mask节点，更新拓扑关系
        """
        for init_name, value in attention_parser.params.items():
            normal_init = normal_subgraph.get_node(prefix + init_name, node_type=OnnxInitializer)
            if normal_init:
                normal_init.value = value

            if init_name.endswith("_perm"):
                transpose_name = init_name[:init_name.index("_perm")]
                transpose_node = normal_subgraph.get_node(prefix+transpose_name, node_type=OnnxNode)
                transpose_node.attrs = {"perm": value}

        for node_name, node in attention_parser.branch_nodes.items():
            input_x = node.inputs[1]
            normal_node = normal_subgraph.get_node(prefix+node_name, node_type=OnnxNode)
            normal_node.inputs[1] = input_x
            if node_name == QK_MASK_ADD and attention_parser.params.get(CONVERT_3DIMS_TO_4DIMS):
                input_x_value = self.graph.get_value_info(input_x)
                if not input_x_value:
                    raise ValueError("Cannot insert expand node without the shape of input {}".format(input_x))

    def replace_subgraph(self, normal_subgraph: OnnxGraph, attention_parser: AttentionParser, prefix):
        """
        用标准子图替换原始的attention子图。
        实现方法：
            用原始attention子图的起始节点和结束节点的拓扑关系，更新标准子图中的起止节点的拓扑关系。
        """
        start_add = normal_subgraph.get_node(prefix + START_ADD, node_type=OnnxNode)
        start_add.inputs = attention_parser.start_node.inputs

        end_add = normal_subgraph.get_node(prefix + END_ADD, node_type=OnnxNode)

        for input_name in attention_parser.end_node.inputs:
            prev_node = attention_parser.graph.get_prev_node(input_name)
            # attention_end其中一个prev_node是attention_start
            if prev_node == attention_parser.start_node:
                end_add.inputs[1] = start_add.outputs[0]
                break

            # atten_end其中一个prev_node不在attention子图中
            if not prev_node:
                end_add.inputs[1] = input_name
                break

        for output_name in attention_parser.end_node.outputs:
            for next_node in self.graph.get_next_nodes(output_name):
                if output_name in next_node.inputs:
                    idx = next_node.inputs.index(output_name)
                    next_node.inputs[idx] = end_add.outputs[0]

    def update_layernorm(self, match_result, ori_shape):
        match_nodes = [node_dict for result in match_result for node_dict in result.node_dicts]
        for i, nodes in enumerate(match_nodes):
            ln_nodes = list(nodes.values())
            first_node = ln_nodes[0][0]
            reshape_name = str(i) + "_layer_norm_reshape"

            # 在第一个layernorm的第一个节点（add, transpose）之后插入reshape，将layer norm以及后面的attention的inputreshape成2维
            # atc的标准pattern要求输入的shape必须得是2维
            if i == 0:
                self.insert_reshape_node(first_node.name, reshape_name, [-1, ori_shape[-1]])

            #在最后一个layer norm的最后一个节点之后插入reshape，将reshape重新reshape原来的shape，否则后面的计算shape会对不上
            if i + 1 == len(match_nodes):
                last_node = ln_nodes[-1][0]
                self.insert_reshape_node(last_node.name, reshape_name, ori_shape)
                if first_node.op_type == "Transpose":
                    transpose_node = self.graph.add_node(name=str(i) + "_transpose", op_type="Transpose",
                                                         attrs=first_node.attrs)
                    self.graph.insert_node(refer_name=reshape_name, insert_node=transpose_node)

            # 有的layer norm的第一个节点是add， 之前的FFN层输出的shape不是2维，需要将shape转成2维
            if first_node.op_type == "Add":
                pre_nodes = [self.graph.get_prev_node(input_x) for input_x in first_node.inputs]
                if len(pre_nodes) != 2:
                    continue
                for prev_node in pre_nodes:
                    if not prev_node or prev_node.op_type != "Reshape":
                        continue
                    reshape_s = self.graph.get_node(prev_node.inputs[1], node_type=OnnxInitializer)
                    if len(reshape_s.value) > 2:
                        reshape_s.value = np.array([-1, reshape_s.value[-1]])

            if first_node.op_type == "Transpose":
                self.graph.remove(first_node.name)

    def update_mask_add_node(self, attention_num):
        for i in range(attention_num):
            prefix = str(i) + "."
            start_add = self.graph.get_node(prefix + START_ADD, node_type=OnnxNode)
            next_nodes = self.graph.get_next_nodes(start_add.outputs[0])
            if len(next_nodes) < 4:
                start_add_b = self.graph.get_node(start_add.inputs[1], node_type=OnnxInitializer)
                unused_add_node_b = self.graph.add_initializer(prefix + "unused_add_node_b",
                                                            np.zeros(start_add_b.value.shape)
                                                               .astype(start_add_b.value.dtype))
                self.graph.add_node(prefix + "unused_add_node", op_type="Add",
                                    inputs=[start_add.outputs[0], unused_add_node_b.name],
                                    outputs=[prefix + "unused_add_node_output"]
                                    )

            qk_mask_add = self.graph.get_node(prefix + QK_MASK_ADD, node_type=OnnxNode)
            input1 = self.graph.get_value_info(qk_mask_add.inputs[0])
            input2 = self.graph.get_value_info(qk_mask_add.inputs[1])
            if input1 and input2:
                if len(input2.shape) < len(input1.shape):
                    diff = len(input1.shape) - len(input2.shape)
                    expand_shape_value = list(copy.deepcopy(input2.shape))
                    for idx in range(diff):
                        expand_shape_value.insert(idx, 1)

                elif len(input2.shape) == len(input1.shape):
                    expand_shape_value = input1.shape
                    expand_shape_value[1] = 1
                else:
                    raise ValueError("The dims of {} is bigger than the dims of {}".format(input2, input1))

                expand_node = self.graph.add_node(name=prefix+"mask_expand", op_type="Expand",
                                                  outputs=[prefix + "mask_expand"])
                self.graph.insert_node(refer_name=qk_mask_add.name, insert_node=expand_node, refer_index=1,
                                       mode="before")
                expand_shape = self.graph.add_initializer(name=prefix + "mask_expand_s",
                                                          value=np.array(expand_shape_value))
                expand_node.inputs.append(expand_shape.name)

            elif input1 and not input2:
                input2 = self.graph.get_node(qk_mask_add.inputs[1], node_type=OnnxInitializer)
                # 给input2的值做广播操作
                if input2.value.shape != input1.shape and input2.value.shape[0] == 1:
                    value = [input2.value[0, :] for _ in range(input1.shape[0])]
                    input2.value = np.array(value)

    def remove_unused_initializers(self):
        all_input = [inp for node in self.graph.nodes for inp in node.inputs]
        all_init = [init.name for init in self.graph.initializers]
        unused_init_names = set(all_init) - set(all_input)
        for init in self.graph.initializers:
            for init_name in unused_init_names:
                if init_name == init.name:
                    self.graph.initializers.remove(init)
                    logger.debug("Remove unused initializer: %s", init_name)


def gen_normal_subgraph(prefix: str = "0."):
    subgraph = OnnxGraph(name="normal_pattern")
    init_value = np.zeros((1, 1))

    add_name = prefix + START_ADD
    add_output_name = add_name + "_output"
    add_node = subgraph.add_node(add_name, op_type="Add", outputs=[add_output_name])

    q_transpose = gen_qkv_block(subgraph, block_input=add_output_name, prefix=prefix)
    k_transpose = gen_qkv_block(subgraph, block_input=add_output_name, block_name="k", prefix=prefix)
    v_transpose = gen_qkv_block(subgraph, block_input=add_output_name, block_name="v", prefix=prefix)

    qk_mm_name = prefix + QK_MATMUL
    qk_mm = subgraph.add_node(
        name=qk_mm_name, op_type="MatMul",
        inputs=[q_transpose.outputs[0], k_transpose.outputs[0]],
        outputs=[qk_mm_name + "_output"]
    )

    mul_name = "bert_" + prefix + MUL  # attentionScore的pattern融合时需要识别mul算子名称以bert_开头
    mul_output_name = mul_name + "_output"
    mul_b = subgraph.add_initializer(name=prefix + MUL_B, value=init_value)
    mul = subgraph.add_node(name=mul_name, op_type="Mul",
                            inputs=[qk_mm.outputs[0], mul_b.name], outputs=[mul_output_name])

    qk_mask_add_name = prefix + QK_MASK_ADD
    qk_mask_add_b = subgraph.add_initializer(name=prefix + QK_MASK_ADD_B, value=init_value)
    qk_mask_add = subgraph.add_node(name=qk_mask_add_name, op_type="Add",
                                    inputs=[mul_output_name, qk_mask_add_b.name],
                                    outputs=[qk_mask_add_name + "_output"])

    softmax_name = prefix + SOFTMAX
    softmax_output_name = softmax_name + "_output"
    softmax = subgraph.add_node(name=softmax_name, attrs={"axis": -1}, op_type="Softmax",
                                inputs=[qk_mask_add.outputs[0]],
                                outputs=[softmax_output_name])

    score_v_mm_name = prefix + SCORE_V_MATMUL
    score_v_mm = subgraph.add_node(name=score_v_mm_name, op_type="MatMul",
                                   inputs=[softmax_output_name, v_transpose.outputs[0]],
                                   outputs=[score_v_mm_name + "_output"])

    transpose_name = prefix + TRANSPOSE
    transpose = subgraph.add_node(name=transpose_name, op_type="Transpose",
                                  attrs={"perm": []},
                                  inputs=[score_v_mm.outputs[0]],
                                  outputs=[transpose_name + "_output"])

    reshape_name = prefix + RESHAPE
    reshape_s = subgraph.add_initializer(name=prefix + RESHAPE_S, value=init_value)
    reshape = subgraph.add_node(name=reshape_name, op_type="Reshape",
                                inputs=[transpose.outputs[0], reshape_s.name],
                                outputs=[reshape_name + "_output"])

    mm_name = prefix + MATMUL
    mm_w = subgraph.add_initializer(name=prefix + MATMUL_W, value=init_value)
    mm = subgraph.add_node(name=mm_name, op_type="MatMul",
                           inputs=[reshape.outputs[0], mm_w.name], outputs=[mm_name + "_output"])

    add_name = prefix + ADD
    add_output = add_name + "_output"
    add_b = subgraph.add_initializer(name=prefix + ADD_B, value=init_value)
    add = subgraph.add_node(name=add_name, op_type="Add",
                            inputs=[mm.outputs[0], add_b.name], outputs=[add_output])

    end_add_name = prefix + END_ADD
    end_add_b = subgraph.add_initializer(name=prefix + END_ADD_B, value=init_value)
    end_add = subgraph.add_node(name=end_add_name, op_type="Add",
                                inputs=[add_output, end_add_b.name],
                                outputs=[end_add_name + "_output"]
                                )
    return subgraph


def gen_qkv_block(subgraph: OnnxGraph, block_input, block_name="q", prefix=""):
    init_value = np.zeros((1, 1))
    init_perm = [1, 1, 1, 1]
    prefix = prefix + block_name + "_"
    if block_name not in ["q", "k", "v"]:
        raise ValueError("Invalid block name {}".format(block_name))

    mm_name = prefix + MATMUL
    mm_w = subgraph.add_initializer(name=prefix + MATMUL_W, value=init_value)
    mm_node = subgraph.add_node(name=mm_name, op_type="MatMul", inputs=[block_input, mm_w.name],
                                outputs=[mm_name + "_output"])

    add_name = prefix + ADD
    add_b = subgraph.add_initializer(name=prefix + ADD_B, value=init_value)
    add_node = subgraph.add_node(add_name, op_type="Add", inputs=[mm_node.outputs[0], add_b.name],
                                 outputs=[add_name + "_output"])

    reshape_name = prefix + RESHAPE
    reshape_s = subgraph.add_initializer(name=reshape_name + "_s", value=init_value)
    reshape_node = subgraph.add_node(name=reshape_name, op_type="Reshape",
                                     inputs=[add_node.outputs[0], reshape_s.name],
                                     outputs=[reshape_name + "_output"]
                                     )

    transpose_name = prefix + TRANSPOSE
    transpose = subgraph.add_node(name=transpose_name, op_type="Transpose", inputs=[reshape_node.outputs[0]],
                                  outputs=[transpose_name + "_output"], attrs={"perm": init_perm})
    end_node = transpose

    if block_name == "k":
        transpose2_name = prefix + "transpose2"
        transpose2_output = transpose2_name + "_output"
        transpose2 = subgraph.add_node(name=transpose2_name, op_type="Transpose", inputs=[transpose.outputs[0]],
                                       outputs=[transpose2_output], attrs={"perm": init_perm}
                                       )
        end_node = transpose2
    return end_node