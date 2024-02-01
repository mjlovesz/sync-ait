# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json

class Node:
    def __init__(self, name = "", type = "", level = 0, order = 0, children = None):
        self.name = name
        self.type = type
        self.level = level
        self.order = order
        self.children = children if children else []
    def __repr__(self):
        return "{} [{}] ({})".format(self.name, self.type, ",".join((x.name for x in self.children)))

    def add_child(self, node):
        self.children.append(node)

class ModelTopo:
    def __init__(self):
        self.root_node = Node("root", "root")

    def create_topo(self, module, module_ids, json_path) -> None:
        _create_sub_topo(module, self.root_node, module_ids)
        _sort_children(self.root_node)
        _tree_to_json(self.root_node, json_path)

    @property
    def get_topo_root(self) -> Node:
        return self.root_node

    @staticmethod
    def json_to_tree(json_path: str) -> Node:
        with open(json_path, "r") as file:
            node_dict = json.loads(file.read(), parse_constant=lambda x: None)
            return _dict_to_tree(node_dict, 0)

    @staticmethod
    def atb_json_to_tree(json_path: str) -> Node:
        with open(json_path, "r") as file:
            node_dict = json.loads(file.read(), parse_constant=lambda x: None)
            return _atb_dict_to_tree(node_dict, 0, 0)

MODULE_ID_NOT_AVAILABLE = -1

def _sort_children(node):
    node.children.sort(key=lambda x: x.order)
    reorder = 0
    for sub_node in node.children:
        if sub_node.order != MODULE_ID_NOT_AVAILABLE:
            sub_node.order = reorder
            reorder = reorder + 1
        _sort_children(sub_node)

def _create_sub_topo(module, node, module_ids):
    new_level = node.level + 1
    for sub_name, sub_module in module.named_children():
        new_name = node.name + "." + sub_name
        new_type = str(type(sub_module).__name__)
        new_order = module_ids.get(new_name, MODULE_ID_NOT_AVAILABLE)
        sub_node = Node(new_name, new_type, new_level, new_order)
        node.add_child(sub_node)
        _create_sub_topo(sub_module, sub_node, module_ids)

def _tree_to_dict(node):
    return {
        "name": node.name,
        "type": node.type,
        "order": node.order,
        "children": [_tree_to_dict(child) for child in node.children]
    }

def _tree_to_json(node, json_path):
    with open(json_path, "w") as file:
        json.dump(_tree_to_dict(node), file)

def _dict_to_tree(node_dict, level):
    node = Node(node_dict["name"], node_dict["type"], level, node_dict["order"])
    for child_dict in node_dict["children"]:
        child_node = _dict_to_tree(child_dict, level + 1)
        node.add_child(child_node)
    return node

def _atb_dict_to_tree(node_dict, level, order):
    if level == 0:
        node = Node("root", "root")
    else:
        node = Node(node_dict["opName"], node_dict["opType"], level, order)
    if "nodes" in node_dict:
        order = 0
        for child_dict in node_dict["nodes"]:
            child_node = _atb_dict_to_tree(child_dict, level + 1, order)
            order = order + 1
            node.add_child(child_node)
    return node
