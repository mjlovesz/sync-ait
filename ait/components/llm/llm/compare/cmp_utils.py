from collections import Counter
from typing import List

from llm.dump.torch_dump.topo import TreeNode

MIN_LAYER_NUMBER = 10


def search_layer_node(root_node: TreeNode):
    child_op_type = [child_node.node_type for child_node in root_node.children]
    if len(child_op_type) > MIN_LAYER_NUMBER:
        op_type_counts = Counter(child_op_type)
        most_count = op_type_counts.most_common(1)[0][1]
        if most_count > MIN_LAYER_NUMBER / 2:
            most_op_type = op_type_counts.most_common(1)[0][0]
            return most_op_type
    else:
        for child_node in root_node.children:
            search_layer_node(child_node)


def get_layer_node(root_node: TreeNode, layer_type: str, layer_nodes: List):
    for child_node in root_node.children:
        if child_node.node_type == layer_type:
            layer_nodes.append(child_node)
        else:
            get_layer_node(child_node, layer_type, layer_nodes)


def get_leaf_nodes(root_node, nodes):
    for child_node in root_node.children:
        if child_node.children:
            get_leaf_nodes(child_node)
        else:
            nodes.append(child_node)
