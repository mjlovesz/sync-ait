# Copyright (c) 2023 Huawei Technologies Co., Ltd.
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

import argparse
import logging
from typing import Dict, List, MutableMapping, Optional, Set, Tuple

import onnx
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, utils


def check_overlapping_names(
        g1: GraphProto, g2: GraphProto, io_map: Optional[List[Tuple[str, str]]] = None
) -> List[Tuple[str, List[str]]]:
    """Checks whether there are name collisions between two graphs

    Returns a list of tuples where the first element represents the member containing overlapping names
    (One of: "node", "edge", "value_info", "initializer", "sparse_initializer"), and the
    second element contains a list of names that appear in both graphs on that category.

    Optionally, it takes an io_map, representing the output/inputs to be connected. It provided, overlapping
    present in the io_map argument will be ignored.
    """
    if type(g1) is not GraphProto:
        raise ValueError("g1 argument is not an ONNX graph")
    if type(g2) is not GraphProto:
        raise ValueError("g2 argument is not an ONNX graph")

    def _overlapping(c1: List[str], c2: List[str]) -> List[str]:
        return list(set(c1) & set(c2))

    def _edge_names(graph: GraphProto, exclude: Optional[Set[str]] = None) -> List[str]:
        if exclude is None:
            exclude = set()
        edges = []
        for node in graph.node:
            for node_input in node.input:
                if node_input != "" and node_input not in exclude:
                    edges.append(node_input)
            for node_output in node.output:
                if node_output != "" and node_output not in exclude:
                    edges.append(node_output)
        return edges

    result = []

    if not io_map:
        io_map = []
    io_map_inputs = {elem[1] for elem in io_map}

    # Edges already cover input/output
    overlap = _overlapping(_edge_names(g1), _edge_names(g2, exclude=io_map_inputs))
    if overlap:
        result.append(("edge", overlap))

    overlap = _overlapping(
        [e.name for e in g1.value_info], [e.name for e in g2.value_info]
    )
    if overlap:
        result.append(("value_info", overlap))

    overlap = _overlapping(
        [e.name for e in g1.initializer], [e.name for e in g2.initializer]
    )
    if overlap:
        result.append(("initializer", overlap))

    overlap = _overlapping(
        [e.values.name for e in g1.sparse_initializer],
        [e.values.name for e in g2.sparse_initializer],
    ) + _overlapping(
        [e.indices.name for e in g1.sparse_initializer],
        [e.indices.name for e in g2.sparse_initializer],
    )
    if overlap:
        result.append(("sparse_initializer", overlap))

    return result


def merge_graphs(  # pylint: disable=too-many-branches,too-many-statements
        g1: GraphProto,
        g2: GraphProto,
        io_map: List[Tuple[str, str]],
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        prefix1: Optional[str] = None,
        prefix2: Optional[str] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
) -> GraphProto:
    """Combines two ONNX graphs into a single one.

    The combined graph is defined by connecting the specified set of outputs/inputs. Those inputs/outputs
    not specified in the io_map argument will remain as inputs/outputs of the combined graph.

    Arguments:
        g1 (GraphProto): First graph
        g2 (GraphProto): Second graph
        io_map (list of pairs of string): The pairs of names [(out0, in0), (out1, in1), ...]
                                            representing outputs of the first graph and inputs of the second
                                            to be connected
        inputs (list of string): Optional list of inputs to be included in the combined graph
                                    By default, all inputs not present in the ``io_map`` argument will be
                                    included in the combined model
        outputs (list of string): Optional list of outputs to be included in the combined graph
                                    By default, all outputs not present in the ``io_map`` argument will be
                                    included in the combined model
        prefix1 (string): Optional prefix to be added to all names in g1
        prefix2 (string): Optional prefix to be added to all names in g2
        name (string): Optional name for the combined graph
                       By default, the name is g1.name and g2.name concatenated with an undescore delimiter
        doc_string (string): Optional docstring for the combined graph
                             If not provided, a default docstring with the concatenation of g1 and g2 docstrings is used

    Returns:
        GraphProto
    """
    if type(g1) is not GraphProto:
        raise ValueError("g1 argument is not an ONNX graph")
    if type(g2) is not GraphProto:
        raise ValueError("g2 argument is not an ONNX graph")

    # Prefixing names in the graph if requested, adjusting io_map accordingly
    if prefix1 or prefix2:
        if prefix1:
            g1_copy = GraphProto()
            g1_copy.CopyFrom(g1)
            g1 = g1_copy
            g1 = add_prefix_graph(g1, prefix=prefix1)
        if prefix2:
            g2_copy = GraphProto()
            g2_copy.CopyFrom(g2)
            g2 = g2_copy
            g2 = add_prefix_graph(g2, prefix=prefix2)
        io_map = [
            (
                prefix1 + io[0] if prefix1 else io[0],
                prefix2 + io[1] if prefix2 else io[1],
            )
            for io in io_map
        ]

    io_map_g1_outs = {io[0] for io in io_map}
    io_map_g2_ins = {io[1] for io in io_map}
    reversed_io_map = {in_name: out_name for out_name, in_name in io_map}
    g1_outs = {g1_onput.name for g1_onput in g1.output}
    g2_ins = {g2_input.name for g2_input in g2.input}

    # If necessary extract subgraphs
    if inputs or outputs:
        if not inputs:
            g1_inputs = [g1_input.name for g1_input in g1.input]
            g2_inputs = [g2_input.name for g2_input in g2.input]
        else:
            input_set = set(inputs)
            g1_inputs = [g1_input.name for g1_input in g1.input if g1_input.name in input_set]
            g2_inputs = [
                g2_input.name
                for g2_input in g2.input
                if g2_input.name in input_set or g2_input.name in io_map_g2_ins
            ]

        if not outputs:
            g1_outputs = [g1_input.name for g1_input in g1.input]
            g2_outputs = [g2_input.name for g2_input in g2.input]
        else:
            output_set = set(outputs)
            g1_outputs = [
                g1_output.name
                for g1_output in g1.output
                if g1_output.name in output_set or g1_output.name in io_map_g1_outs
            ]
            g2_outputs = [g2_output.name for g2_output in g2.output if g2_output.name in output_set]

        if len(g1_inputs) < len(g1.input) or len(g1_outputs) < len(g1.output):
            e1 = utils.Extractor(helper.make_model(g1))
            g1 = e1.extract_model(g1_inputs, g1_outputs).graph

        if len(g2_inputs) < len(g2.input) or len(g2_outputs) < len(g2.output):
            e2 = utils.Extractor(helper.make_model(g2))
            g2 = e2.extract_model(g2_inputs, g2_outputs).graph

    # Check that input/output names specified in the io_map argument are valid input/output names
    for g1_out_name, g2_in_name in io_map:
        if g1_out_name not in g1_outs:
            raise ValueError(f"Output {g1_out_name} is not present in g1")
        if g2_in_name not in g2_ins:
            raise ValueError(f"Input {g2_in_name} is not present in g2")

    # Check for name collision
    overlapping_names = check_overlapping_names(g1, g2, io_map)
    if len(overlapping_names) > 0:
        category, names = overlapping_names[0]
        raise ValueError(
            "Cant merge two graphs with overlapping names. "
            f"Found repeated {category} names: "
            + ", ".join(names)
            + "\n"
            + "Consider using ``onnx.compose.add_prefix`` to add a prefix to names in one of the graphs."
        )

    g = GraphProto()

    g.node.extend(g1.node)
    g2_nodes_begin = len(g.node)
    g.node.extend(g2.node)
    g2_nodes_end = len(g.node)

    # Connecting outputs of the first graph with the inputs of the second
    for node_idx in range(g2_nodes_begin, g2_nodes_end):
        node = g.node[node_idx]
        for index, name_ in enumerate(node.input):
            if name_ in reversed_io_map:
                node.input[index] = reversed_io_map[name_]

    if inputs:
        input_set = set(inputs)
        g.input.extend([g1_input for g1_input in g1.input if g1_input.name in input_set])
        g.input.extend([g2_input for g2_input in g2.input if g2_input.name in input_set])
    else:
        g.input.extend(g1.input)
        g.input.extend([g2_input for g2_input in g2.input if g2_input.name not in io_map_g2_ins])

    if outputs:
        output_set = set(outputs)
        g.output.extend([g1_output for g1_output in g1.output if g1_output.name in output_set])
        g.output.extend([g2_output for g2_output in g2.output if g2_output.name in output_set])
    else:
        g.output.extend([g1_output for g1_output in g1.output if g1_output.name not in io_map_g1_outs])
        g.output.extend(g2.output)

    g.initializer.extend(g1.initializer)
    g.initializer.extend(
        [init for init in g2.initializer if init.name not in io_map_g2_ins]
    )

    g.sparse_initializer.extend(g1.sparse_initializer)
    g.sparse_initializer.extend(
        [
            init
            for init in g2.sparse_initializer
            if init.values.name not in io_map_g2_ins
        ]
    )

    g.value_info.extend(g1.value_info)
    g.value_info.extend([vi for vi in g2.value_info if vi.name not in io_map_g2_ins])

    g.name = name if name is not None else "_".join([g1.name, g2.name])

    if doc_string is None:
        doc_string = (
                f"Graph combining {g1.name} and {g2.name}\n"
                + g1.name
                + "\n\n"
                + g1.doc_string
                + "\n\n"
                + g2.name
                + "\n\n"
                + g2.doc_string
        )
    g.doc_string = doc_string

    return g


def merge_models(  # pylint: disable=too-many-branches
        m1: ModelProto,
        m2: ModelProto,
        io_map: List[Tuple[str, str]],
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        prefix1: Optional[str] = None,
        prefix2: Optional[str] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
        producer_name: Optional[str] = "onnx.compose.merge_models",
        producer_version: Optional[str] = "1.0",
        domain: Optional[str] = "",
        model_version: Optional[int] = 1,
) -> ModelProto:
    """Combines two ONNX models into a single one.

    The combined model is defined by connecting the specified set of outputs/inputs.
    Those inputs/outputs not specified in the io_map argument will remain as
    inputs/outputs of the combined model.

    Both models should have the same IR version, and same operator sets imported.

    Arguments:
        m1 (ModelProto): First model
        m2 (ModelProto): Second model
        io_map (list of pairs of string): The pairs of names [(out0, in0), (out1, in1), ...]
                                          representing outputs of the first graph and inputs of the second
                                          to be connected
        inputs (list of string): Optional list of inputs to be included in the combined graph
                                 By default, all inputs not present in the ``io_map`` argument will be
                                 included in the combined model
        outputs (list of string): Optional list of outputs to be included in the combined graph
                                  By default, all outputs not present in the ``io_map`` argument will be
                                  included in the combined model
        prefix1 (string): Optional prefix to be added to all names in m1
        prefix2 (string): Optional prefix to be added to all names in m2
        name (string): Optional name for the combined graph
                       By default, the name is g1.name and g2.name concatenated with an undescore delimiter
        doc_string (string): Optional docstring for the combined graph
                             If not provided, a default docstring with the concatenation of g1 and g2 docstrings is used
        producer_name (string): Optional producer name for the combined model. Default: 'onnx.compose'
        producer_version (string): Optional producer version for the combined model. Default: "1.0"
        domain (string): Optional domain of the combined model. Default: ""
        model_version (int): Optional version of the graph encoded. Default: 1

    Returns:
        ModelProto
    """
    if type(m1) is not ModelProto:
        raise ValueError("m1 argument is not an ONNX model")
    if type(m2) is not ModelProto:
        raise ValueError("m2 argument is not an ONNX model")

    if m1.ir_version != m2.ir_version:
        raise ValueError(
            f"IR version mismatch {m1.ir_version} != {m2.ir_version}."
            " Both models should have the same IR version"
        )
    ir_version = m1.ir_version

    opset_import_map: MutableMapping[str, int] = {}
    opset_imports = list(m1.opset_import) + list(m2.opset_import)

    for entry in opset_imports:
        if entry.domain in opset_import_map:
            found_version = opset_import_map.get(entry.domain, default=None)
            if entry.version != found_version:
                raise ValueError(
                    "Can't merge two models with different operator set ids for a given domain. "
                    f"Got: {m1.opset_import} and {m2.opset_import}"
                )
        else:
            opset_import_map[entry.domain] = entry.version

    # Prefixing names in the graph if requested, adjusting io_map accordingly
    if prefix1 or prefix2:
        if prefix1:
            m1_copy = ModelProto()
            m1_copy.CopyFrom(m1)
            m1 = m1_copy
            m1 = add_prefix(m1, prefix=prefix1)
        if prefix2:
            m2_copy = ModelProto()
            m2_copy.CopyFrom(m2)
            m2 = m2_copy
            m2 = add_prefix(m2, prefix=prefix2)
        io_map = [
            (
                prefix1 + io[0] if prefix1 else io[0],
                prefix2 + io[1] if prefix2 else io[1],
            )
            for io in io_map
        ]
    merge_graphs_info = MergeGraphsInfo(io_map=io_map, inputs=inputs, outputs=outputs)
    graph = merge_graphs(
        m1.graph,
        m2.graph,
        io_map=io_map,
        inputs=inputs,
        outputs=outputs,
        name=name,
        doc_string=doc_string,
    )
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version,
        opset_imports=opset_imports,
        ir_version=ir_version,
    )

    # Merging model metadata props
    model_props = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props.get(meta_entry.key, default=None)
            if value != meta_entry.value:
                raise ValueError(
                    "Can't merge models with different values for the same model metadata property."
                    f" Found: property = {meta_entry.key}, with values {value} and {meta_entry.value}."
                )
        else:
            model_props[meta_entry.key] = meta_entry.value
    helper.set_model_props(model, model_props)

    # Merging functions
    function_overlap = list(
        {f.name for f in m1.functions} & {f.name for f in m2.functions}
    )
    if function_overlap:
        raise ValueError(
            "Can't merge models with overlapping local function names."
            " Found in both graphs: " + ", ".join(function_overlap)
        )
    model.functions.MergeFrom(m1.functions)
    model.functions.MergeFrom(m2.functions)

    # remove original checker: checker.check_model(model)
    return model


def add_prefix_graph(  # pylint: disable=too-many-branches
        graph: GraphProto,
        prefix: str,
        rename_nodes: Optional[bool] = True,
        rename_edges: Optional[bool] = True,
        rename_inputs: Optional[bool] = True,
        rename_outputs: Optional[bool] = True,
        rename_initializers: Optional[bool] = True,
        rename_value_infos: Optional[bool] = True,
        inplace: Optional[bool] = False,
        name_map: Optional[Dict[str, str]] = None,
) -> GraphProto:
    """Adds a prefix to names of elements in a graph: nodes, edges, inputs, outputs,
    initializers, sparse initializer, value infos.

    It can be used as a utility before merging graphs that have overlapping names.
    Empty names are not prefixed.

    Arguments:
        graph (GraphProto): Graph
        prefix (str): Prefix to be added to each name in the graph
        rename_nodes (bool): Whether to prefix node names
        rename_edges (bool): Whether to prefix node edge names
        rename_inputs (bool): Whether to prefix input names
        rename_outputs (bool): Whether to prefix output names
        rename_initializers (bool): Whether to prefix initializer and sparse initializer names
        rename_value_infos (bool): Whether to prefix value info names
        inplace (bool): If True, mutates the graph directly.
                        Otherwise, a copy will be created
        name_map: (Dict): shared name_map in subgraph

    Returns:
        GraphProto
    """
    if type(graph) is not GraphProto:
        raise ValueError("graph argument is not an ONNX graph")

    if not inplace:
        g = GraphProto()
        g.CopyFrom(graph)
    else:
        g = graph

    def _prefixed(prefix: str, name: str) -> str:
        return prefix + name if len(name) > 0 else name

    if name_map is None:
        name_map = {}
    if rename_edges:
        for g_node in g.node:
            for e in g_node.input:
                name_map[e] = _prefixed(prefix, e)
            for e in g_node.output:
                name_map[e] = _prefixed(prefix, e)

    if rename_inputs:
        for entry in g.input:
            name_map[entry.name] = _prefixed(prefix, entry.name)
    if rename_outputs:
        for entry in g.output:
            name_map[entry.name] = _prefixed(prefix, entry.name)

    if rename_nodes:
        for g_node in g.node:
            g_node.name = _prefixed(prefix, g_node.name)
            for attribute in g_node.attribute:
                if attribute.g:
                    add_prefix_graph(
                        attribute.g, prefix, inplace=True, name_map=name_map
                    )

    if rename_initializers:
        for init in g.initializer:
            name_map[init.name] = _prefixed(prefix, init.name)
        for sparse_init in g.sparse_initializer:
            name_map[sparse_init.values.name] = _prefixed(
                prefix, sparse_init.values.name
            )
            name_map[sparse_init.indices.name] = _prefixed(
                prefix, sparse_init.indices.name
            )

    if rename_value_infos:
        for entry in g.value_info:
            name_map[entry.name] = _prefixed(prefix, entry.name)

    for g_node in g.node:
        for idx, output in enumerate(g_node.output):
            if g_node.output[idx] in name_map:
                g_node.output[idx] = name_map.get(output, default=None)
        for idx, input_ in enumerate(g_node.input):
            if g_node.input[idx] in name_map:
                g_node.input[idx] = name_map.get(input_, default=None)

    for in_desc in g.input:
        if in_desc.name in name_map:
            in_desc.name = name_map.get(in_desc.name, default=None)
    for out_desc in g.output:
        if out_desc.name in name_map:
            out_desc.name = name_map.get(out_desc.name, default=None)

    for initializer in g.initializer:
        if initializer.name in name_map:
            initializer.name = name_map.get(initializer.name, default=None)
    for sparse_initializer in g.sparse_initializer:
        if sparse_initializer.values.name in name_map:
            sparse_initializer.values.name = name_map.get(sparse_initializer.values.name, default=None)
        if sparse_initializer.indices.name in name_map:
            sparse_initializer.indices.name = name_map.get(sparse_initializer.indices.name, default=None)

    for value_info in g.value_info:
        if value_info.name in name_map:
            value_info.name = name_map.get(value_info.name, default=None)

    return g


def add_prefix(
        model: ModelProto,
        prefix: str,
        rename_nodes: Optional[bool] = True,
        rename_edges: Optional[bool] = True,
        rename_inputs: Optional[bool] = True,
        rename_outputs: Optional[bool] = True,
        rename_initializers: Optional[bool] = True,
        rename_value_infos: Optional[bool] = True,
        rename_functions: Optional[bool] = True,
        inplace: Optional[bool] = False,
) -> ModelProto:
    """Adds a prefix to names of elements in a graph: nodes, edges, inputs, outputs,
    initializers, sparse initializer, value infos, and local functions.

    It can be used as a utility before merging graphs that have overlapping names.
    Empty names are not _prefixed.

    Arguments:
        model (ModelProto): Model
        prefix (str): Prefix to be added to each name in the graph
        rename_nodes (bool): Whether to prefix node names
        rename_edges (bool): Whether to prefix node edge names
        rename_inputs (bool): Whether to prefix input names
        rename_outputs (bool): Whether to prefix output names
        rename_initializers (bool): Whether to prefix initializer and sparse initializer names
        rename_value_infos (bool): Whether to prefix value info nanes
        rename_functions (bool): Whether to prefix local function names
        inplace (bool): If True, mutates the model directly.
                        Otherwise, a copy will be created

    Returns:
        ModelProto
    """
    if type(model) is not ModelProto:
        raise ValueError("model argument is not an ONNX model")

    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m

    add_prefix_graph(
        model.graph,
        prefix,
        rename_nodes=rename_nodes,
        rename_edges=rename_edges,
        rename_inputs=rename_inputs,
        rename_outputs=rename_outputs,
        rename_initializers=rename_initializers,
        rename_value_infos=rename_value_infos,
        inplace=True,  # No need to create a copy, since it's a new model
    )

    if rename_functions:
        f_name_map = {}
        for f in model.functions:
            new_f_name = prefix + f.name
            f_name_map[f.name] = new_f_name
            f.name = new_f_name
        # Adjust references to local functions in other local function
        # definitions
        for f in model.functions:
            for n in f.node:
                if n.op_type in f_name_map:
                    n.op_type = f_name_map.get(n.op_type, default=None)
        # Adjust references to local functions in the graph
        for n in model.graph.node:
            if n.op_type in f_name_map:
                n.op_type = f_name_map.get(n.op_type, default=None)

    return model


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description="Merge two ONNX model into one")
parser.add_argument("--previous_model_path", type=str, help="Path to the previous ONNX model")
parser.add_argument("--following_model_path", type=str, help="Path to the following ONNX model")
parser.add_argument("--merge_model_path", type=str, help="Path to the merged ONNX model")
parser.add_argument("--previous_model_outputs", type=str, nargs='+', help="Output name of previous model")
parser.add_argument("--following_model_inputs", type=str, nargs='+', help="Input name of following model")

args = parser.parse_args()
previous_model_path = args.previous_model_path
following_model_path = args.following_model_path
merge_model_path = args.merge_model_path
previous_model_outputs = args.previous_model_outputs
following_model_inputs = args.following_model_inputs

len_previous_model_outputs = len(previous_model_outputs)
len_following_model_inputs = len(following_model_inputs)
if len_previous_model_outputs != len_following_model_inputs:
    raise Exception("Each input and output must match!")

connection_list = []
for i in range(len_previous_model_outputs):
    connection = (previous_model_outputs[i], following_model_inputs[i])
    connection_list.append(connection)
    logging.info("connect %s and %s ", previous_model_outputs[i], following_model_inputs[i])

previous_model = onnx.load(previous_model_path)
following_model = onnx.load(following_model_path)

max_ir_version = max(previous_model.ir_version, following_model.ir_version)
previous_model.ir_version = max_ir_version
following_model.ir_version = max_ir_version

combined_model = merge_models(
    previous_model, following_model,
    io_map=connection_list
)
onnx.save(combined_model, merge_model_path)
logging.info("merge %s and % s into %s", previous_model_path, previous_model_path, merge_model_path)
