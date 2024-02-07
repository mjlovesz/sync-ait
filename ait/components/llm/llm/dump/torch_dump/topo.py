# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import stat
import json

FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
MODULE_ID_NOT_AVAILABLE = -1


class TreeNode:
    def __init__(self, node_name: str, node_type: str, level=0, order=0):
        self.node_name = node_name
        self.node_type = node_type
        self.level = level
        self.order = order
        self.children = []

    def __repr__(self):
        return "{} [{}] ({})".format(self.node_name, self.node_type, ",".join((x.node_name for x in self.children)))

    def add_child(self, node):
        self.children.append(node)

    def sort_children(self):
        self.children.sort(key=lambda x: x.order)
        reorder = 0
        for sub_node in self.children:
            if sub_node.order != MODULE_ID_NOT_AVAILABLE:
                sub_node.order = reorder
                reorder = reorder + 1
            sub_node.sort_children()


class ModelTree:
    def __init__(self):
        self.root_node = TreeNode("root", "root")

    def create_tree(self, module, module_ids, json_path) -> None:
        self.root_node.node_type = str(type(module).__name__)
        self._create_sub_tree(module, self.root_node, module_ids)
        self.root_node.sort_children()
        _tree_to_json(self.root_node, json_path)

    @staticmethod
    def json_to_tree(json_path: str) -> TreeNode:
        with open(json_path, "r") as file:
            node_dict = json.loads(file.read(), parse_constant=lambda x: None)
            return _dict_to_tree(node_dict, 0, 0)

    @staticmethod
    def atb_json_to_tree(json_path: str) -> TreeNode:
        with open(json_path, "r") as file:
            node_dict = json.loads(file.read(), parse_constant=lambda x: None)
            return _atb_dict_to_tree(node_dict, 0, 0)

    def _create_sub_tree(self, module, father_node, module_ids):
        new_level = father_node.level + 1
        for sub_name, sub_module in module.named_children():
            new_name = father_node.node_name + "." + sub_name
            new_type = str(type(sub_module).__name__)
            new_order = module_ids.get(new_name, MODULE_ID_NOT_AVAILABLE)
            sub_node = TreeNode(new_name, new_type, new_level, new_order)
            father_node.add_child(sub_node)
            self._create_sub_tree(sub_module, sub_node, module_ids)


def _tree_to_dict(node):
    return {
        "name": node.node_name,
        "type": node.node_type,
        "children": [_tree_to_dict(child) for child in node.children],
    }


def _tree_to_json(node, json_path):
    with os.fdopen(os.open(json_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), 'w') as file:
        json.dump(_tree_to_dict(node), file)


def _dict_to_tree(node_dict, level, order):
    node = TreeNode(node_dict["name"], node_dict["type"], level, order)
    sub_level = level + 1
    sub_order = 0
    for child_dict in node_dict["children"]:
        child_node = _dict_to_tree(child_dict, sub_level, sub_order)
        node.add_child(child_node)
        sub_order = sub_order + 1
    return node


def _atb_dict_to_tree(node_dict, level, order):
    if level == 0:
        node = TreeNode("root", node_dict["modelName"])
    else:
        node = TreeNode(node_dict["opName"], node_dict["opType"], level, order)
    if "nodes" in node_dict:
        reorder = 0
        for child_dict in node_dict["nodes"]:
            child_node = _atb_dict_to_tree(child_dict, level + 1, reorder)
            reorder = reorder + 1
            node.add_child(child_node)
    return node
