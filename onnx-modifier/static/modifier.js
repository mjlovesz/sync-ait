var modifier = modifier || {};
// op - revertop table
var cmd_map = {
    'add_node' : 'delete_node',
    'delete_node' : 'recover_node',
    'add_input' : 'delete_input',
    'delete_input' : 'add_input',
    'add_output' : 'delete_output',
    'delete_output' : 'add_output',
    'change_prop' : 'change_prop',
    'delete_child' : 'add_child',
    'add_child' : 'delete_child',
    'recover_node' : 'delete_node',
    'change_ori_ini' : 'change_ori_ini',
    'change_add_ini' : 'change_add_ini',
    'change_node_attr' : 'change_node_attr',
    'change_input_size' : 'change_input_size',
    'change_node_io' : 'change_node_io'
}

modifier.Modifier = class {
    constructor(view) {
        this.view = view;
        this.model = null;
        this.graphs = null;
        this.name2ModelNode = new Map();
        this.name2ViewNode = new Map();
        this.name2NodeStates = new Map();
        this.namedEdges = new Map();

        this.oriInputs = new Set();
        this.cmd_list = new Array();
        this.idx = -1;
        this.addedOutputs = new Set();
        this.addedInputs = new Set();
        this.addedNode = new Map();
        this.addNodeKey = 0;
        this.changedAttributes = new Map();
        this.initializerEditInfo = new Map();
        this.renameMap = new Map();
        this.reBatchInfo = new Map();
        this.inputSizeInfo = new Map();

        this.downloadWithShapeInf = false;
        this.downloadWithCleanUp = false;

        this.modelProperties = new Map();
        this.extract_start = new Set();
        this.extract_end = new Set();
        this.extract_highlight_nodes = [];
    }

    loadModelGraph(model, graphs) {
        this.model = model;
        this.graphs = graphs;
        this.graph = this.graphs[0];
        // this.analyzeModelGraph();
        for (const input_name of this.graph.inputs) {
            this.oriInputs.add(input_name)
        }
        this.updateAddNodeDropDown();
        this.resetGraph()
    }

    updateAddNodeDropDown() {
        // update dropdown supported node lost
        var addNodeDropdown = this.view._host.document.getElementById('add-node-dropdown');
        for (const node of this.model.supported_nodes) {
            // node: [domain, op]
            var option = new Option(node[1], node[0] + ':' + node[1]);
            // console.log(option)
            addNodeDropdown.appendChild(option);
        }
    }

    undo() {
        if (this.idx == -1) return
        var op = this.cmd_list[this.idx]
        var un_op_type = cmd_map[op[0]], op_content = op[1], un_content = op[2];
        switch(un_op_type) {

            // delete node only when add node by user
            case 'delete_node' : 
            this.addedNode.delete(un_content[0])
            this.addNodeKey --
            this.deleteSingleNode(un_content[0], false)
            this.idx --
            break

            // delete output when add output by user
            case 'delete_output' : 
            this.addedOutputs.delete(un_content[0])
            this.deleteModelOutput('out_' + un_content[0], false)
            this.idx --
            break

            // delete input when add input by user
            case 'delete_input' : 
            this.addedInputs.delete(un_content[0])
            this.deleteModelInput(un_content[0], false)
            this.idx --
            break

            // add output when output is deleted
            case 'add_output' :
            // set visible
            this.name2NodeStates.set(un_content[0], 'Exist');
            // check whether original output
            var flag = false
            for (const ori_output of this.graph._outputs) {
                const model_name = ori_output.modelNodeName
                if (model_name == un_content[0]) {
                    flag = true
                    break
                }
            }

            // not original output, update addedoutputs
            if (!flag) {
                this.addedOutputs.add(un_content[0].substring(4))// out_ + xxx
            }
            this.idx --
            this.applyAndUpdateView();
            break

            // add input when delete input
            case 'add_input':
            // set input visible
            this.name2NodeStates.set(un_content[0], 'Exist')
            // check whether originial input
            var flag = false
            for (const ori_input of this.graph._inputs) {
                const model_name = ori_input.modelNodeName
                if (model_name == un_content[0]) {
                    flag = true
                    break
                }
            }
            // not originial input, update addedinputs
            if (!flag){
                this.addedInputs.add(un_content[0])
            }
            this.idx --
            this.applyAndUpdateView()
            break

            // recover node when delete node happened
            case 'recover_node':
            this.recoverSingleNode(un_content[0], false)
            this.idx --
            break

            // delete node if recover node happened
            case 'delete_node':
            this.deleteSingleNode(un_content[0], false)
            break

            // delete child if recover child happened
            case 'delete_child':
            this.deleteNodeWithChildren(un_content[0], false)
            break

            // add child when delete child happened
            case 'add_child':
            this.recoverNodeWithChildren(un_content[0], false)
            this.idx --
            break

            // set input size to previous value when input size is changed
            case 'change_input_size':
            this.changeInputSize(un_content[0], un_content[1], [], false)
            this.idx --
            this.applyAndUpdateView();
            break

            // set original node initializer to previous value when initializer is changed
            case 'change_ori_ini':
            // if previous is none
            if (un_content[1] == undefined) {
                this.initializerEditInfo.set(un_content[0], [])
            }
            else {
                this.initializerEditInfo.set(un_content[0], un_content[1])
            }
            this.idx --
            this.applyAndUpdateView();
            break

            // set added node initializer to previous value when initializer is changed
            case 'change_add_ini':
            // if previous value is none
            if (un_content[1] == undefined) {
                this.initializerEditInfo.set(un_content[0], [])
            }
            else {
                this.initializerEditInfo.set(un_content[0], un_content[1])
            }
            this.idx --
            this.applyAndUpdateView();
            break

            // set model properties to previous value when properties is changed
            case 'change_prop':
            this.changeModelProperties(un_content[0], un_content[1], un_content[2], false)
            this.idx --
            this.applyAndUpdateView();
            break

            // set node input/output to previous value when node input/output is changed
            case 'change_node_io':
            this.changeNodeInputOutput(un_content[0], un_content[1], un_content[2], un_content[3],
                un_content[4], un_content[5], false)
            this.idx --
            break

            // set node attribute to previous value when node attribute is changed
            case 'change_node_attr':
            var modelNodeName = un_content[0], attributeName = un_content[1], ori_value = un_content[2], type = un_content[3]
            this.changeNodeAttribute(modelNodeName, attributeName, ori_value, type, false)
            this.idx --
            break
            default:
            break
        }
    }

    redo() {
        // if at the latest op, can't redo
        if(this.idx == this.cmd_list.length - 1) return
        //index move forward
        this.idx ++
        var op = this.cmd_list[this.idx]
        var op_type = op[0], op_content = op[1], un_content = op[2];
        // redo op according to the op content
        switch(op_type) {
            case 'add_node':
            this.addNode(op_content[0], op_content[1],false)
            break
            case 'add_output':
            this.addModelOutput(op_content[0], false)
            break
            case 'add_input':
            this.addModelInput(op_content[0], false)
            break
            case 'delete_node':
            this.deleteSingleNode(op_content[0], false)
            break
            case 'delete_output':
            this.deleteModelOutput(op_content[0], false)
            break
            case 'delete_input':
            this.deleteModelInput(op_content[0], false)
            break
            case 'delete_child':
            this.deleteNodeWithChildren(op_content[0], false)
            break
            case 'change_input_size':
            this.changeInputSize(op_content[0], op_content[1], [], false)
            this.applyAndUpdateView();
            break
            case 'change_ori_ini':
            this.changeInitializer(op_content[0], op_content[1], op_content[2], op_content[3], op_content[4], op_content[5], op_content[6], false)
            this.applyAndUpdateView();
            break
            case 'change_add_ini':
            this.changeAddedNodeInitializer(op_content[0], op_content[1], op_content[2], op_content[3], op_content[4], op_content[5], op_content[6], false)
            this.applyAndUpdateView();
            break
            case 'change_prop':
            this.changeModelProperties(op_content[0], op_content[1], op_content[2], false)
            this.applyAndUpdateView();
            break
            case 'change_node_attr':
            this.changeNodeAttribute(op_content[0], op_content[1], op_content[2], op_content[3], false)
            break
            case 'change_prop':
            this.changeModelProperties(op_content[0], op_content[1], op_content[2], false)
            this.applyAndUpdateView();
            break
            case 'change_node_io':
            this.changeNodeInputOutput(op_content[0], op_content[1], op_content[2], op_content[3], op_content[4], op_content[5], false)
            break
            default:
            break
        }
    }

    // ======= Record modified info =======> //
    addNode(op_domain, op_type, is_valid=true) {
        var node_id = (this.addNodeKey++).toString();  // in case input (onnx) node has no name
        var modelNodeName = 'custom_added_' + op_type + node_id;

        var properties = new Map();
        properties.set('domain', op_domain);
        properties.set('op_type', op_type);
        properties.set('name', modelNodeName);
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['add_node', [op_domain, op_type], [modelNodeName]])
        }
        this.addedNode.set(modelNodeName, new view.LightNodeInfo(properties));
        this.name2NodeStates.set(modelNodeName, 'Exist');
        this.applyAndUpdateView();
    }

    addModelOutput(node_name, is_valid=true) {
        var modelNode = this.name2ModelNode.get(node_name);
        // use a output argument as a proxy
        let name2NodeOutput = 'out_' + modelNode.outputs[0].arguments[0].name
        if (this.name2NodeStates.has(name2NodeOutput)) {
            this.name2NodeStates.set(name2NodeOutput, 'Exist');
        }
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['add_output', [node_name], [modelNode.outputs[0].arguments[0].name]])
            
        }
        this.addedOutputs.add(modelNode.outputs[0].arguments[0].name);
        this.applyAndUpdateView();
    }
    addModelInput(input_name, is_valid=true) {
        // use a input argument as a proxy
        // this.addedInputs.add(modelNode.inputs[0].arguments[0].name);
        if (this.name2NodeStates.has(input_name)) {
            this.name2NodeStates.set(input_name, 'Exist');
        }
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (!this.oriInputs.has(input_name)) {
            if (is_valid) {
                this.cmd_list.length = this.idx + 1
                this.cmd_list.push(['add_input', [input_name], [input_name]])
                this.idx ++
            }
            this.addedInputs.add(input_name)
        }
        this.applyAndUpdateView();
    }

    deleteModelOutput(output_name, is_valid=true) {
        this.name2NodeStates.set(output_name, 'Deleted');  // "out_" + xxx
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['delete_output', [output_name], [output_name]])
        }
        this.applyAndUpdateView();
    }

    deleteModelInput(input_name, is_valid=true) {
        this.name2NodeStates.set(input_name, 'Deleted');  // "out_" + xxx
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['delete_input', [input_name], [input_name]])
        }
        this.applyAndUpdateView();
    }

    changeModelProperties(prop_name, prop_value, index=undefined, is_valid=true) {
        if (index !== undefined) {
            if (!this.modelProperties.has(prop_name)) {
                this.modelProperties.set(prop_name, [])
            }
            var ori_value = this.modelProperties.get(prop_name)[index]
            // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
            if (is_valid) {
                this.cmd_list.length = this.idx + 1
                this.idx ++
                this.cmd_list.push(['change_prop', [prop_name, prop_value, index], [prop_name, ori_value, index]])
            }
            this.modelProperties.get(prop_name)[index] = prop_value
        } else {
            var pre_value = this.modelProperties.get(prop_name)
            this.modelProperties.set(prop_name, prop_value)
            // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
            if (is_valid) 
            {this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['change_prop', [prop_name, prop_value, index], [prop_name, pre_value, index]])}
        }
    }
    
    setExtractStart(node_name, is_start) {
        if (is_start) {
            this.extract_start.add(node_name)
        } else {
            this.extract_start.delete(node_name)
        }
        this.highLightExtractNodes()
    }
    
    setExtractEnd(node_name, is_end) {
        if (is_end) {
            this.extract_end.add(node_name)
        } else {
            this.extract_end.delete(node_name)
        }
        this.highLightExtractNodes()
    }
    
    getExtractStart() {
        return this.extract_start
    }

    getExtractEnd() {
        return this.extract_end
    }

    highLightExtractNodes() {
        let start_nodes = this.getExtractStart()
        let end_nodes = this.getExtractEnd()

        let inside_nodes = this.getInsideNodes(start_nodes, end_nodes)

        for (const ori_node of this.extract_highlight_nodes) {
            this.name2ViewNode.get(ori_node).element.classList.remove("graph-node-extract-node")
            this.name2ViewNode.get(ori_node).element.classList.remove("graph-node-extract-end")
            this.name2ViewNode.get(ori_node).element.classList.remove("graph-node-extract-start")
        }
        this.extract_highlight_nodes = []

        for (const inside_node of inside_nodes) {
            this.name2ViewNode.get(inside_node).element.classList.add("graph-node-extract-node")
            this.extract_highlight_nodes.push(inside_node)
        }
        
        for (const start_node of start_nodes) {
            this.name2ViewNode.get(start_node).element.classList.add("graph-node-extract-start")
            this.extract_highlight_nodes.push(start_node)
        }
        for (const end_node of end_nodes) {
            this.name2ViewNode.get(end_node).element.classList.add("graph-node-extract-end")
            this.extract_highlight_nodes.push(end_node)
        }
    }

    getInsideNodes(start_nodes, end_nodes) {
        if (!start_nodes || !end_nodes) {
            return []
        }
        if (start_nodes == end_nodes) {
            return []
        }

        let cached_node = new Map()
        let reach_node = new Set()
        
        for (const start_node of start_nodes) {
            this.is_reach_end_node(start_node, end_nodes, cached_node, reach_node)
        }

        return [...reach_node]
    }

    is_reach_end_node(this_node_name, end_nodes, cached_node, reach_node) {
        if (cached_node.has(this_node_name)) {
            return cached_node.get(this_node_name)
        }

        if (end_nodes.has(this_node_name)) {
            reach_node.add(this_node_name)
        }

        if (!this.namedEdges.has(this_node_name)) {
            return false
        }

        for (const next_node of this.namedEdges.get(this_node_name)) {
            let is_next_reach = this.is_reach_end_node(next_node, end_nodes, cached_node, reach_node)
            cached_node.set(next_node, is_next_reach)
            if (is_next_reach) {
                reach_node.add(this_node_name)
            }
        }
        return reach_node.has(this_node_name)
    }

    clickSingleNode(node_name) {
        this.name2ViewNode.get(node_name).element.classList.add("graph-node-clicked")
    }

    clearHighlightNode() {
        for (const elem of document.getElementsByClassName("graph-node-highlight")) {
            elem.classList.remove("graph-node-highlight")
        }
    }

    deleteSingleNode(node_name, is_valid=true) {
        this.name2NodeStates.set(node_name, 'Deleted');
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['delete_node', [node_name], [node_name]])
        }
        this.applyAndUpdateView();
    }

    deleteNodeWithChildren(node_name, is_valid=true) {
        if (this.name2NodeStates.get(node_name) == 'Deleted') return;

        this.name2NodeStates.set(node_name, 'Deleted');

        if (!this.namedEdges.has(node_name)) return; // for leaf node

        for (var i = 0; i < this.namedEdges.get(node_name).length; i++) {
            this.deleteNodeWithChildren(this.namedEdges.get(node_name)[i]);
        }
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['delete_child', [node_name], [node_name]])
        }
        this.applyAndUpdateView();
    }

    recoverSingleNode(node_name, is_valid=true) {
        this.name2NodeStates.set(node_name, 'Exist');
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['recover_node', [node_name], [node_name]])
        }
        this.applyAndUpdateView();
    }

    recoverNodeWithChildren(node_name, is_valid=true) {
        if (this.name2NodeStates.get(node_name) == 'Exist') return;

        this.name2NodeStates.set(node_name, 'Exist');

        if (!this.namedEdges.has(node_name)) return; // for leaf node

        for (var i = 0; i < this.namedEdges.get(node_name).length; i++) {
            this.recoverNodeWithChildren(this.namedEdges.get(node_name)[i]);
        }
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['add_child', [node_name], [node_name]])
        }
        this.applyAndUpdateView();
    }

    changeNodeInputOutput(modelNodeName, parameterName, param_type, param_index, arg_index, targetValue, is_valid=true) {
        var arg_name = ""
        if (this.addedNode.has(modelNodeName)) {  // for custom added node 
            if (this.addedNode.get(modelNodeName).inputs.has(parameterName)) {
                arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];  // [arg.name, arg.is_optional]
                // update the corresponding initializer name
                if (this.initializerEditInfo.has(arg_name)) {
                    var init_val = this.initializerEditInfo.get(arg_name);
                    this.initializerEditInfo.set(targetValue, init_val);
                    this.initializerEditInfo.delete(arg_name);
                }
                this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0] = targetValue;
            }
            // console.log(this.initializerEditInfo)

            if (this.addedNode.get(modelNodeName).outputs.has(parameterName)) {
                arg_name = this.addedNode.get(modelNodeName).outputs.get(parameterName)[arg_index][0]
                this.addedNode.get(modelNodeName).outputs.get(parameterName)[arg_index][0] = targetValue;
            }
            // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
            if (is_valid) {
                this.cmd_list.length = this.idx + 1
                this.idx ++
                this.cmd_list.push(['change_node_io', [modelNodeName, parameterName, param_type, param_index, arg_index, targetValue], [modelNodeName, parameterName, param_type, param_index, arg_index, arg_name]])
            }
        }
        // console.log(this.addedNode)

        else {    // for the nodes in the original model
            var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
            // console.log(orig_arg_name)

            if (!this.renameMap.get(modelNodeName)) {
                this.renameMap.set(modelNodeName, new Map());
            }
            this.renameMap.get(modelNodeName).set(orig_arg_name, targetValue);
            // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
            if (is_valid) {
                this.cmd_list.length = this.idx + 1
                this.idx ++
                this.cmd_list.push(['change_node_io', [modelNodeName, parameterName, param_type, param_index, arg_index, targetValue], [modelNodeName, parameterName, param_type, param_index, arg_index, orig_arg_name]])
            }
            // console.log(this.modifier.renameMap)
        }

        this.applyAndUpdateView();
    }
    

    getOriginalName(param_type, modelNodeName, param_index, arg_index) {
        if (param_type == 'model_input') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).arguments[0].original_name;
        }

        if (param_type == 'model_output') {
            // modelNodeName = 'out_' + modelNodeName
            // console.log(modelNodeName)
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).arguments[0].original_name;
            // console.log(orig_arg_name)
        }

        if (param_type == 'input') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).inputs[param_index].arguments[arg_index].original_name;
            // console.log(orig_arg_name)
        }
        if (param_type == 'output') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).outputs[param_index].arguments[arg_index].original_name;
            // console.log(orig_arg_name)
        }

        return orig_arg_name;
    }

    changeInitializer(modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue, is_valid=true) {
        var arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
        var ori_value = this.initializerEditInfo.get(arg_name)
        this.initializerEditInfo.set(arg_name, [type, targetValue]);
        // this.view._updateGraph()
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['change_ori_ini', [modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue], [arg_name, ori_value]])
        }
        this.applyAndUpdateView();
    }

    changeAddedNodeInitializer(modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue, is_valid=true) {
        var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];
        var ori_value = this.initializerEditInfo.get(arg_name)
        this.initializerEditInfo.set(arg_name, [type, targetValue]);
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['change_add_ini', [modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue], [arg_name, ori_value]])
        }
        // this.view._updateGraph()

        this.applyAndUpdateView();
    }

    changeNodeAttribute(modelNodeName, attributeName, targetValue, type, is_valid=true) {
        var ori_value = undefined, ori_type = undefined
        if (this.addedNode.has(modelNodeName)) {
            ori_value, ori_type = this.addedNode.get(modelNodeName).attributes.get(attributeName)
            this.addedNode.get(modelNodeName).attributes.set(attributeName, [targetValue, type]);
        }
        // console.log(this._addedNode)

        else {    // for the nodes in the original model
            if (!this.changedAttributes.get(modelNodeName)) {
                this.changedAttributes.set(modelNodeName, new Map());
            }
            var node = this.name2ModelNode.get(modelNodeName);
            for (var i = 0; i < node._attributes.length; ++i) {
                if (attributeName == node._attributes[i].name) {
                    // [val, type]
                    ori_value = node._attributes[i]._value
                    ori_type = node._attributes[i]._type
                }
            }
            this.changedAttributes.get(modelNodeName).set(attributeName, [targetValue, type]);
        }

        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['change_node_attr', [modelNodeName, attributeName, targetValue, type], [modelNodeName, attributeName, ori_value, ori_type]])
        }
        // this.view._updateGraph()
        this.applyAndUpdateView();
    }

    changeInputSize(input_name, value, ori_value, is_valid=true) {
        this.inputSizeInfo.set(input_name, value)
        // if is operated by user, add to cmd_list([op_type, op_content, revertop_content])
        if (is_valid) {
            this.cmd_list.length = this.idx + 1
            this.idx ++
            this.cmd_list.push(['change_input_size', [input_name, value], [input_name, ori_value]])
        }
    }

    onOffShapeInf(turnedOn) {
        if (turnedOn)  this.downloadWithShapeInf = true;
        else this.downloadWithShapeInf = false;
    }

    onOffCleanUp(turnedOn) {
        if (turnedOn)  this.downloadWithCleanUp= true;
        else this.downloadWithCleanUp = false;
    }
    // <======= Record modified info ======= //

    // ======= Apply modified info and update view =======> //
    deleteEnter() {
        this.applyAndUpdateView();
    }

    refreshModelInputOutput() {
        // console.log(this.modifier.renameMap)
        for (var input of this.graph._inputs) {
            var input_orig_name = input.arguments[0].original_name;
            if (this.renameMap.get(input_orig_name)) {
                var new_name = this.renameMap.get(input_orig_name).get(input_orig_name);
                var arg_with_new_name = this.graph._context.argument(new_name, input_orig_name);

                input.arguments[0] = arg_with_new_name;

                // change all the name of node input linked with model input meanwhile
                for (var node of this.graph._nodes) {
                    for (var node_input of node.inputs) {
                        for (const [index, element] of node_input.arguments.entries()) {
                            if (element.original_name == input_orig_name) {
                                var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                                node_input.arguments[index] = arg_with_new_name;

                                // save the changed name into _renameMap
                                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                                if (!this.renameMap.get(node.modelNodeName)) {
                                    this.renameMap.set(node.modelNodeName, new Map());
                                }

                                var orig_arg_name = element.original_name;
                                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
                            }
                        }
                    }
                }
            }
        }
        // console.log(this.graph.outputs)
        // create and add new output to graph
        this.graph.reset_custom_modified_outputs();
        for (var output_name of this.addedOutputs) {
            this.graph.add_output(output_name);
        }
        for (var input_name of this.addedInputs) {
            this.graph.add_input(input_name, this.inputSizeInfo.get(input_name));
        }
        for (let [name, size_info] of this.inputSizeInfo) {
            this.graph.modify_input_shape(name, size_info);
        }
        // console.log(this.graph.outputs)
        for (var output of this.graph.outputs) {
            var output_orig_name = output.arguments[0].original_name;
            if (this.renameMap.get('out_' + output_orig_name)) {
                // for model input and output, node.modelNodeName == element.original_name
                var new_name = this.renameMap.get('out_' + output_orig_name).get(output_orig_name);
                // console.log(new_name)
                var arg_with_new_name = this.graph._context.argument(new_name, output_orig_name);

                output.arguments[0] = arg_with_new_name;

                // change all the name of node output linked with the model output meanwhile
                for (var node of this.graph._nodes) {
                    for (var node_output of node.outputs) {
                        for (const [index, element] of node_output.arguments.entries()) {
                            if (element.original_name == output_orig_name) {
                                // console.log(element.original_name)
                                var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                                node_output.arguments[index] = arg_with_new_name;

                                // save the changed name into _renameMap
                                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                                if (!this.renameMap.get(node.modelNodeName)) {
                                    this.renameMap.set(node.modelNodeName, new Map());
                                }

                                var orig_arg_name = element.original_name;
                                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
                            }
                        }
                    }
                }
            }
        }
        
        for (var output of this.graph.outputs) {
            var output_orig_name = output.arguments[0].original_name;
            if (this.name2NodeStates.get('out_' + output_orig_name) == "Deleted") {
                this.graph.delete_output(output_orig_name);
            }
        }
        
        for (var input_info of this.graph.inputs) {
            var input_orig_name = input_info.arguments[0].original_name;
            if (this.name2NodeStates.get(input_orig_name) == "Deleted") {
                this.graph.delete_input(input_orig_name);
            }
        }
    }

    // re-generate the added node according to addedNode according to the latest addedNode
    refreshAddedNode() {
        this.graph.reset_custom_added_node();
        // for (const node_info of this.addedNode.values()) {
        // for (const [modelNodeName, node_info] of this.lastViewGraph.addedNode) {
        for (const [modelNodeName, node_info] of this.addedNode) {
            // console.log(node_info)
            var node = this.graph.make_custom_added_node(node_info);
            // console.log(node)

            for (const input of node.inputs) {
                var arg_list_info = [];
                for (const arg of input._arguments) {
                    arg_list_info.push([arg.name, arg.is_optional]);
                }
                this.addedNode.get(modelNodeName).inputs.set(input.name, arg_list_info);
            }

            for (const output of node.outputs) {
                var arg_list_info = [];
                for (const arg of output._arguments) {
                    arg_list_info.push([arg.name, arg.is_optional]);
                }
                this.addedNode.get(modelNodeName).outputs.set(output.name, arg_list_info);
            }

        }
    }

    // re-fresh node arguments in case the node inputs/outputs are changed
    refreshNodeArguments() {
        for (var node of this.graph._nodes) {
            // if (this.modifier.renameMap.get(node.modelNodeName)) {
            if (this.renameMap.get(node.modelNodeName)) {

                // check inputs
                for (var input of node.inputs) {
                    for (const [index, element] of input.arguments.entries()) {
                        if (this.renameMap.get(node.modelNodeName).get(element.original_name)) {
                            var new_name = this.renameMap.get(node.modelNodeName).get(element.original_name);
                            var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                            input.arguments[index] = arg_with_new_name;
                        }
                    }
                }

                // check outputs
                for (var output of node.outputs) {
                    for (const [index, element] of output.arguments.entries()) {
                        if (this.renameMap.get(node.modelNodeName).get(element.original_name)) {
                            var new_name = this.renameMap.get(node.modelNodeName).get(element.original_name);
                            // console.log(new_name)
                            var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                            output.arguments[index] = arg_with_new_name;
                        }
                    }
                }
            }
        }
    }

    refreshNodeAttributes() {
        for (const node_name of this.changedAttributes.keys()) {
            var attr_change_map = this.changedAttributes.get(node_name);
            var node = this.name2ModelNode.get(node_name);

            for (var i = 0; i < node._attributes.length; ++i) {
                if (attr_change_map.get(node._attributes[i].name)) {
                    // [val, type]
                    node._attributes[i]._value = attr_change_map.get(node._attributes[i].name)[0];
                }
            }
        }
    }

    resetGraph() {
        // reset node states
        for (const name of this.name2NodeStates.keys()) {
            this.name2NodeStates.set(name, 'Exist');
        }

        for (const name of this.addedInputs) {
            this.name2NodeStates.delete(name)
        }
        for (const name of this.addedOutputs) {
            this.name2NodeStates.delete('out_' + name)
        }

        // console.log(this.modifier.renameMap)
        // reset node inputs/outputs
        for (const changed_node_name of this.renameMap.keys()) {
            var node = this.name2ModelNode.get(changed_node_name);
            // console.log(node)
            // console.log(typeof node)
            // console.log(node.constructor.name)
            if (node.arguments) {   // model input or model output. Because they are purely onnx.Parameter
                // node.arguments[0] = this.graph._context.argument(node.modelNodeName);
                node.arguments[0] = this.graph._context.argument(node.arguments[0].original_name);
            }

            else {                   // model nodes
                //reset inputs
                for (var input of node.inputs) {
                    for (var i = 0; i < input.arguments.length; ++i) {
                        // console.log(input.arguments[i].original_name)
                        if (this.renameMap.get(node.modelNodeName).get(input.arguments[i].original_name)) {
                            input.arguments[i] = this.graph._context.argument(input.arguments[i].original_name);
                        }
                    }
                }

                // reset outputs
                for (var output of node.outputs) {
                    for (var i = 0; i < output.arguments.length; ++i) {
                        if (this.renameMap.get(node.modelNodeName).get(output.arguments[i].original_name)) {
                            output.arguments[i] = this.graph._context.argument(output.arguments[i].original_name);
                        }
                    }
                }

            }
        }
        this.renameMap = new Map();
        this.changedAttributes = new Map();
        this.reBatchInfo = new Map();
        this.inputSizeInfo = new Map();
        this.initializerEditInfo = new Map();

        // clear custom added nodes
        this.addedNode = new Map();
        this.graph.reset_custom_added_node();
        this.addedOutputs = new Set();
        this.addedInputs = new Set();
        this.graph.reset_custom_modified_outputs();

        // reset load location
        var container = this.view._getElementById('graph');
        container.scrollLeft = 0;
        container.scrollTop = 0;
        this.view._zoom = 1;

        this.extract_start = new Set();
        this.extract_end = new Set();
        this.highLightExtractNodes()
        this.extract_highlight_nodes = [];

        this.applyAndUpdateView();
    }

    applyAndUpdateView() {
        this.refreshAddedNode();
        this.refreshModelInputOutput();
        this.refreshNodeArguments();
        this.refreshNodeAttributes();

        // this.graphs has been modified (inplace)
        this.view._updateGraph(this.model, this.graphs);
    }
    // <======= Apply modified info and update view ======= //

}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Modifier = modifier.Modifier;
}

