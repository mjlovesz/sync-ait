from collections import Counter
from typing import List

from llm.dump.torch_dump.topo import TreeNode

MIN_LAYER_NUMBER = 10


def search_layer_node(root_node: TreeNode):
    layer_node_type = ""

    def run(node):
        nonlocal layer_node_type
        if layer_node_type:
            return
        child_op_type = [child_node.op_type for child_node in node.children]
        if len(child_op_type) > MIN_LAYER_NUMBER:
            op_type_counts = Counter(child_op_type)
            most_count = op_type_counts.most_common(1)[0][1]
            if most_count > MIN_LAYER_NUMBER / 2:
                most_op_type = op_type_counts.most_common(1)[0][0]
                layer_node_type = most_op_type
        else:
            for child_node in node.children:
                run(child_node)

    run(root_node)
    return layer_node_type


def get_layer_node(root_node: TreeNode, layer_type: str):
    all_layer_nodes = []

    def run(node, layer_type, layer_nodes):
        for child_node in node.children:
            if child_node.op_type == layer_type:
                layer_nodes.append(child_node)
            else:
                run(child_node, layer_type, layer_nodes)

    run(root_node, layer_type, all_layer_nodes)
    return all_layer_nodes


def get_leaf_nodes(root_node):
    all_leaf_nodes = []

    def run(node, leaf_nodes):
        for child_node in node.children:
            if child_node.children:
                run(child_node, leaf_nodes)
            else:
                leaf_nodes.append(child_node)

    run(root_node, all_leaf_nodes)
    return all_leaf_nodes


def get_all_nodes(root_node):
    all_nodes = [root_node]

    def run(node, children_nodes):
        for child_node in node.children:
            children_nodes.append(child_node)
            if child_node.children:
                run(child_node, children_nodes)

    run(root_node, all_nodes)
    return all_nodes