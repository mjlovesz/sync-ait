
// Experimental

var om = om || {};
var svp = {};
var protobuf = protobuf || require('./protobuf');
var base = base || require('./base');

function formatEnumToString(i) {
    const FORMAT_LIST = ["NCHW", "NHWC", "ND", "NC1HWC0", "Fractal-Z", "NC1C0HW Pad", "NHWC1C0", "FSR_NCHW",
                        "Fractal-Deconv", "C1HWNC0", "Fractal-Deconv-Transposed", "Fractal-Deconv-SP-Stride-Trans",
                        "NC1HWC0(C04)", "Fractal-Z(C04)", "CHWN", "Fractal-Deconv-SP-Stride8-Trans", "HWCN",
                        "NC1KHKWHWC0", "BN", "Filter HWCK"];
    return FORMAT_LIST[i];
}

function flopsToString(flops) {
    let result = new Number(flops / 1e6).toFixed(1) + "M";
    if (flops >= 1e12) {
        result = new Number(flops / 1e12).toFixed(1) + "T";
    } else if (flops > 1e9) {
        result = new Number(flops / 1e9).toFixed(1) + "G";
    }
}

function isNodeConst(op) {
    return op.type == "Const" || op.type == "QuantizedConst";
}

om.ModelFactory = class {

    match(context) {
        return om.Container.open(context);
    }

    open(context, match) {
        return om.Metadata.open(context).then((metadata) => {
            if (match === 'IMOD' || match === 'CUST') {
                var target = new om.Container(context, match);
                return context.require('./om-proto').then(() => {
                    try {
                        target._loadmodel(context, false);
                    } catch (error) {
                        target._loadModel(context, true);
                    }
                    return new om.Model(metadata, [target.model], [target.weights], target.device);
                });
            } else {
                throw new om.Error('Unsupported DaVinci OM ' + this.signature + ' signature.');
            }
        });
    }
};

om.Container = class {

    static open(context) {
        const stream = context.stream;
        if (stream && stream.length >= 256) {
            const buffer = stream.peek(4);
            const signature = Array.from(buffer).map((c) => String.fromCharCode(c)).join('');
            if (signature === 'IMOD' || signature === 'CUST') {
                return signature;
            }
        }
        return null;
    }

    constructor(context, signature) {
        this._context = context;
        this._signature = signature;
        this.device = new Map();
        if (signature == 'IMOD') {
            this._offset = 0;
        } else if (signature == 'CUST') {
            var reader = new base.BinaryReader(context.stream);
            this._offset = reader.view.getUint32(4, true) + 16;
        }
    }

    _loadModel(context, isHugeModel) {
        const MODEL_DEF = 0;
        const MODEL_WEIGHT = 1;
        const MODEL_TASK = 2;
        const MODEL_SIGNATURE = 3;
        const MODEL_UNKNOWN1 = 4;
        const MODEL_UNKNOWN2 = 8;
        const MODEL_UNKNOWN3 = 238;
        const DEVICE_CONFIG = 5;
        const HEADER_SIZE = 256;
        var TABLE_INDEX_SIZE;
        var MODEL_PARTITION_MEM_INFO_SIZE;
        if (isHugeModel) {
            TABLE_INDEX_SIZE = 8;
            MODEL_PARTITION_MEM_INFO_SIZE = 24;
        } else {
            TABLE_INDEX_SIZE = 4;
            MODEL_PARTITION_MEM_INFO_SIZE = 12;
        }
        const stream = context.stream;
        const reader = new base.BinaryReader(stream);
        const buffer = reader.read(4);
        this.format = 'DaVinci OM';
        const decoder = new TextDecoder('utf-8');
        const size = reader.uint32();
        this.version = reader.uint32();
        this.checksum = reader.read(64);
        reader.skip(4);
        this.is_encrypt = reader.byte();
        this.is_checksum = reader.byte();
        this.type = reader.byte(); // 0=IR model, 1=standard model, 2=OM Tiny model
        this.mode = reader.byte(); // 0=offline, 1=online
        this.name = decoder.decode(reader.read(32));
        this.ops = reader.uint32();
        this.userdefineinfo = reader.read(32);
        this.ir_version = reader.uint32();
        this.model_num = reader.uint32();
        this.platform_version = reader.read(20);
        this.platform_type = reader.byte();
        reader.seek(0);
        reader.skip(size); // skip(HAEDER_SIZE)
        var partitions;
        if (isHugeModel) {
            partitions = new Array(reader.uint64());
            for (let i = 0; i < partitions.length; i++) {
                partitions[i] = {
                    type: reader.uint64(),
                    offset: reader.uint64(),
                    size: reader.uint64()
                };
            }
        } else {
            partitions = new Array(reader.uint32());
            for (let i = 0; i < partitions.length; i++) {
                partitions[i] = {
                    type: reader.uint32(),
                    offset: reader.uint32(),
                    size: reader.uint32()
                };
            }
        }
        const offset = this._offset + HEADER_SIZE + TABLE_INDEX_SIZE + MODEL_PARTITION_MEM_INFO_SIZE * partitions.length;
        for (const partition of partitions) {
            reader.seek(offset + partition.offset);
            const buffer = reader.read(partition.size);
            switch (partition.type) {
                case MODEL_DEF: { // MODEL_DEF
                    this.model = buffer;
                    break;
                }
                case MODEL_WEIGHT: { // WEIGHTS_DATA
                    this.weights = buffer;
                    break;
                }
                case MODEL_TASK: {// TASK_INFO
                    break;
                }
                case MODEL_SIGNATURE: { // TBE_KERNELS
                    break;
                }
                case MODEL_UNKNOWN1: { // CUST_AICPU_KERNELS
                    break;
                }
                case MODEL_UNKNOWN2: {
                    break;
                }
                case MODEL_UNKNOWN3: {
                    break;
                }
                case DEVICE_CONFIG: { // DEVICE_CONFIG
                    let reader = new base.BinaryReader(this._context.stream);
                    let content = reader.view;
                    const decoder = new TextDecoder('ascii');
                    try {
                        let position = 4;
                        while (position < partition.size) {
                            let length = content.getUint32(offset - this._offset + position, true);
                            reader.seek(0);
                            reader.skip(position + 4);
                            let buffer = reader.read(length);
                            let name = decoder.decode(buffer);
                            let device = content.getUint32(offset - this.offset + position + 4 + length, true);
                            position += length + 8;
                            this.devices.set(name, device);
                        }
                    } catch {
                        // Ignore if failed to parse device config
                    }
                    break;
                }
                default: {
                    throw new om.Error("Unsupported partition type '" + partition.type + "'.");
                }
            }
        }
        if (!this.model.length) {
            throw new om.Error('File does not contain a model definition.');
        }
        try{
            om.proto = protobuf.get('om').om;
            const omReader = protobuf.BinaryReader.open(this.model);
            this.model = om.proto.ModelDef.decode(omReader);
        } catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new om.Error('File format is not ge.proto.ModelDef (' + message.replace(/\.$/, '') + ').');
        }
    }
};

om.Metadata = class {
    static open(context) {
        om.Metadata.textDecoder = om.Metadata.textDecoder || new TextDecoder('utf-8');
        if (om.Metadata._metadata) {
            return Promise.resolve(om.Metadata._metadata);
        }
        return context.request('./om-metadata.json', 'utf-8', null).then((data) => {
            om.Metadata._metadata = new om.Metadata(data);
            return om.Metadata._metadata;
        }).catch(() => {
            om.Metadata._metadata = new om.Metadata(null);
            return om.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (let item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                        this._map[item.name].attributes.push({"name": "format", "type": "Format"});
                    }
                }
            }
        }
    }

    type(type) {
        var nodeType = this._map[type] || null;
        if (nodeType) {
            nodeType["name"] = type;
        }
        return nodeType;
    }

    attribute(type, name) {
        let map = this._attributeCache[type];
        if (!map) {
            map = {};
            const schema = this.type(type);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (let attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[type] = map;
        }
    }
};

om.Model = class {

    constructor(metadata, nets, weights, deviceConfigMap) {
        this._deviceConfigMap = deviceConfigMap;
        this._weights = weights;
        this._format = "DaVinci OM";
        this._flops = 0;
        this._npuFlops = 0;
        this._graphs = [];
        this.format = target.format;

        for (let i = 0; i < nets.length; ++i) {
            let index = nets.length == 1 ? undefined : i+1;
            let net = nets[i];
            let mainGraph = net.graph[0];
            let weight = this._weights[i];

            if (nets.length > 1) {
                this._modelType = "Multi-shape model";
            } else {
                if ("graph_infershaped_flag" in mainGraph.attr && mainGraph.attr["graph_infershaped_flag"].b) {
                    this._modelType = "Shape Infered By NPU IR Model";
                } else if ("ir_infershaped" in mainGraph.attr && mainGraph.attr["ir_infershaped"].b) {
                    this._modelType = "Shape Infered By DDK IR Model";
                } else if ("hiai_version" in mainGraph.attr && new TextDecoder("utf-8").decode(mainGraph.attr["hiai_version"].s) == "ir") {
                    this._modelType = "Shape Uninfered IR Model";
                } else if ("memory_size" in mainGraph.attr) {
                    this._modelType = `Compiled Model`;
                    this._description = `memory: ${mainGraph.attr["memory_size"].i}`;
                    if ("weight_size" in mainGraph.attr) {
                        this._description |= `, weight: ${mainGraph.attr["weight_size"].i}`;
                    }
                } else if ("memory_size" in net.attr) {
                    this._modelType = `Legacy Model`;
                    this._description = `memory: ${net.attr["memory_size"].i}, weight: ${net.attr["weight_size"].i}`;

                } else {
                    this._modelType = `Unknown format`;
                }
            }

            for (var j = 0; j < net.graph.length; ++j) {
                this._extractGraph(metadata, net.graph[j], net, this, weight, index, "");
            }
        }
    }

    _extractGraph(metadata, graph, net, model, weight, index, parentName) {
        for (let op of graph.op) {
            for (let item in op.attr) {
                if (Object.prototype.hasOwnProperty.call(op.attr[item], 'g')) {
                    let subgraph = op.attr[item].g;
                    if (index != undefined) {
                        subgraph.name = subgraph.name + "-shape: " + index;
                    }
                    this._extractGraph(metadata, subgraph, net, model, weight, index, subgraph.name);
                    delete op.attr[item];
                } else if (item == "subgraph" && Object.prototype.hasOwnProperty.call(op.attr[item], 'bt')) {
                    let sb = op.attr[item].bt;
                    let subgraph = om.proto.GraphDef.decode(sb);
                    subgraph.name = parentName + "/" + subgraph.name;
                    this._extractGraph(metadata, subgraph, net, model, weight, index, subgraph.name);
                }
            }

            if (this._deviceConfigMap.has(op.name)) {
                let config = this._deviceConfigMap.get(op.name);
                var candidate_devices = [];
                if (config & 0x1) {
                    candidate_devices.push("NPU");
                }
                if ((config >> 1) & 0x1) {
                    candidate_devices.push("CPU");
                }
                op.attr["device"] = candidate_devices.join(" && ");
            }
        }
        if (index != undefined) {
            graph.name = graph.name + "-shape: " + index;
        }
        this._graphs.unshift(new om.Graph(metadata, graph, weight, model));
    }

    get format() {
        return this._format || "DaVinci OM";
    }

    get _modelType() {
        return this._modelType || "DaVinci OM";
    }

    get producer() {
        return this._producer || '';
    }

    set flops(v) {
        this._flops = v;
    }

    set _npuFlops(v) {
        this._npuFlops = v;
    }

    get flops() {
        return this._flops;
    }

    get npuFlops() {
        return this._npuFlops;
    }

    get description() {
        if (!this._description) {
            return `FLOPs: ${flopsToString(this.flops)} NPU FLOPs: ${flopsToString(this.npuFlops)}`;
        }
        return `FLOPs: ${flopsToString(this.flops)} NPU FLOPs: ${flopsToString(this.npuFlops)}, ${this._description}`;
    }

    get source() {
        return this._source || '';
    }

    get graphs() {
        return this._graphs;
    }
};

om.Graph = class {

    constructor(context, graph) {
        this._model = model;
        this._node = [];
        this._inputs = [];
        this._outputs = [];
        this._name = graph.name;
        this._weight = weight;
        this._flops = 0;
        this._npuFlops = 0;
        var mainGraph = graph;
        for (var op of mainGraph.op) {
            if (!isNodeConst(op)) {
                op.name = (op.name == "") ? "internal_unnamed" : op.name;
                this._nodes.push(new om.Node(metadata, op, mainGraph, this._weight, model));
            }
        }
    }

    get name() {
        return this._name;
    }

    get groups() {
        return false;
    }

    get nodes() {
        return this._nodes;
    }

    get outputs() {
        return this._outputs;
    }

    get inputs() {
        return this._inputs;
    }
};

om.Node = class {
    static enum2Dtype(val) {
        const TYPE_LIST = ["undefined", "float32", "float16", "int8", "uint8", "int16", "uint16", "int32",
                            "uint32", "int64", "uint64", "bool", "float64"];
        return TYPE_LIST[val];
    }
    static enum2DtypeInner(val) {
        const TYPE_LIST = ["float32", "float16", "int8", "int32", "uint8", "", "int16", "uint16", "uint32",
                            "int64", "uint64", "float64", "bool", "dual", "dual int8", "dual uint8"];
        return TYPE_LIST[val];
    }
    constructor(context, op, graph, value, tensors) {
        this._model = model;
        this._name = op.name;
        this._weight = weight;
        this._metadata = metadata;
        this._type = metadata.type(op.type) || { name: op.type };
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];
        this._chains = [];
        this._controlDependencies = [];
        this._device = null;

        const ATTR_BLACK_LIST = [];

        var schema = metadata.type(this._type.name);

        if (!schema) {
            schema = metadata.type("Undefined");
        }

        let inputIdx = 0; //The length of input might not equal to input_desc, empty items in input refer to optional inputs, which are not in input_desc.
        let weightDims = null;
        for (let i = 0; i < op.input.length; ++i) {
            if (op.input[i] == "") {
                continue;
            }
            let pos = op.input[i].lastIndexOf(":");
            let name = (pos == 0) ? "internal_unnamed" : op.input[i].slice(0, pos);
            var src_index = op.input[i].slice(pos+1);
            if (src_index == -1) {
                this._controlDependencies.push(name);
                continue;
            }
            let schemaName = (i < schema.inputs.length) ? schema.inputs[i].name : schema.inputs[schema.inputs.length-1].name;
            let inputNode = graph.op.find(node => node.name == name);
            let inputFormat = op.input_desc[inputIdx].layout;

            if (isNodeConst(inputNode)) {
                var inputDims = null;
                if (inputNode.attr["value"].t.desc.shape != null) {
                    inputDims = inputNode.attr["value"].t.desc.shape.dim;
                }
                // For a compiled model, the shape may be a raw shape or a fractal-z shape. The following lines ensure getting the raw shape.
                if ('origin_shape' in inputNode.attr["value"].t.desc.attr) {
                    inputDims = inputNode.attr["value"].t.desc.attr["origin_shape"].list.i;
                }
                let inputDtype = om.Node.enum2Dtype(inputNode.attr["value"].t.desc.dtype);
                if (!weightDims) { //Save the first const as weight
                    weightDims = inputDims;
                }

                var data = null;

                if (inputNode.attr["value"].t.data == '') {
                    if (this._weight == null) {
                        data = null;
                    } else if ('merged_offset' in inputNode.attr['value'].t.desc.attr) {
                        let offset = inputNode.attr['value'].t.desc.attr['merged_offset'].i;
                        data = this._weight.slice(offset, offset + inputNode.attr['value'].t.desc.weight_size);
                    } else {
                        let offset = inputNode.attr['value'].t.desc.data_offset;
                        data = this._weight.slice(offset, offset + inputNode.attr['value'].t.desc.weight_size);
                    }
                } else {
                    data = inputNode.attr["value"].t.data;
                }
                let datalength = (data == null) ? 0 : data.length;
                let tensor = new om.Argument(name,
                    null,
                    new om.Tensor('Constant', new om.TensorType(inputDtype, inputDims, inputFormat, inputNode.attr['value'].t.desc.layout, datalength), data)
                );
                this._inputs.push(new om.Parameter(schemaName, true, [tensor]));
            } else {
                let inputDims = op.input_desc[inputIdx].shape ? op.input_desc[inputIdx].shape.dim : undefined;
                let inputDtype = op.input_desc[i] ? om.Node.enum2Dtype(op.input_desc[i].dtype) : "undefined";
                let inputName = src_index == 0 ? name : name + ":" + src_index;
                this._inputs.push(new om.Parameter(schemaName, true, [new om.Argument(inputName, new om.TensorType(inputDtype, inputDims, inputFormat, null), null)]));
            }
            ++inputIdx;
        }

        var outputIdx = 0;
        for (let outputDesc of op.output_desc) {
            let outputDims = outputDesc.shape ? outputDesc.shape.dim : undefined;
            let outputDtype = om.Node.enum2Dtype(outputDesc.dtype);
            let outputFormat = outputDesc.layout;
            let outputName = (outputIdx == 0) ? this._name : this._name + ":" + outputIdx;
            let outputSchemaName = outputIdx < schema.outputs.length ? schema.outputs[outputIdx].name : schema.outputs[schema.output.length-1].name;
            this._outputs.push(new om.Parameter(outputSchemaName, true, [new om.Argument(outputName, new om.TensorType(outputDtype, outputDims, outputFormat), null)]));
            ++outputIdx;
        }

        for (var attr in op.attr) {
            if (!ATTR_BLACK_LIST.includes(attr)) {
                var value = op.attr[attr];
                if (attr == "device") {
                    this._device = value;
                    continue;
                }
                if (Object.prototype.hasOwnProperty.call(value, 'func')) {
                    let attrInFunc = this._extractGraph(value.func, attr+".");
                    for (let [k, v] of attrInFunc) {
                        this._attributes.push(new om.Attribute(null, k, v, schema, true));
                    }
                    continue;
                }
                this._attributes.push(new om.Attribute(null, attr, value, schema, true));
            }
        }

        if (op.type == "Convolution" || op.type == "ConvolutionDepthwise" || op.type == "ConvTranspose") {
            let outputDims = op.output_desc[0].shape ? op.output_desc[0].shape.dim : undefined;
            let inputDims = op.input_desc[0].shape ? op.input_desc[0].shape.dim : undefined;
            let format = "Unknown";

            if ("format" in op.attr) {
                format = formatEnumToString(op.attr["format"].i);
            } else if ("data_format" in op.attr) {
                format = new TextDecoder("utf-8").decode(op.attr["data_format"].s);
            } else if ("NHWC_FORMAT" in op.attr) {
                format = formatEnumToString(op.attr["NHWC_FORMAT"].b) ? "NHWC" : "NCHW";
            } else if (op.input_desc[0].layout != undefined) {
                format = op.input_desc[0].layout;
            }
            let group = 1;
            if ("groups" in op.attr) {
                group = op.attr["group"].i;
            }

            if (inputDims == undefined || outputDims == undefined) {
                format = "Unknown shape";
            }

            weightDims = op.input_desc[1].shape.dim; // To confirm the format of filter

            var ho = 0;
            var wo = 0;
            var co = 0;
            var ci = 0;
            var kh = 0;
            var kw = 0;
            var hi = 0;
            var wi = 0;

            var strideH = 0;
            var strideW = 0;
            if (op.attr["stride"] != null) {
                strideH = op.attr["strides"].list.i[0];
                strideW = op.attr["strides"].list.i[1];
            } else {
                strideH = op.attr["stride"].list.i[0];
                strideW = op.attr["stride"].list.i[1];
            }

            if (format == "NCHW") {
                ho = outputDims[2];
                wo = outputDims[3];
                co = outputDims[1];
                ci = inputDims[1];
                hi = inputDims[2];
                wi = inputDims[3];
                kh = weightDims[2];
                kw = weightDims[3];
            } else if (format == "NHWC" ) {
                ho = outputDims[1];
                wo = outputDims[2];
                co = outputDims[3];
                ci = inputDims[3];
                hi = inputDims[1];
                wi = inputDims[2];
                kh = weightDims[2];
                kw = weightDims[3];
            }
            let M = ho * wo;
            if (op.type == "Convolution") {
                let N = co;
                let K = kh * kw * ci;
                let flops = M * N * K * 2 / group;
                this._attributes.push(new om.Attribute(null, "FLOPs (CPU)", flopsToString(flops), schema, true));
                model.flops = model.flops + flops;

                let K_ = kh * kw * (~~((ci + 15) / 16)) * 16;
                let N_ = (~~((N + 15) / 16)) * 16;
                let flops_ = M * N_ * K_ * 2 / group;
                this._attributes.push(new om.Attribute(null, "FLOPs (NPU)", flopsToString(flops_), schema, true));
                model.npuFlops = model.npuFlops + flops_;
            } else if (op.type == "ConvolutionDepthwise") {
                let N = 1;
                let K = kh * kw;
                let flops = ci * M * N * K *2;
                this._attributes.push(new om.Attribute(null, "FLOPs (CPU)", flopsToString(flops_), schema, true));
                model.flops = model.flops + flops_;
                let flops_ = (~~((ci + 15) / 16)) * 16 * M * N * K * 2;
                this._attributes.push(new om.Attribute(null, "FLOPs (NPU)", flopsToString(flops_), schema, true));
                model.npuFlops = model.npuFlops + flops_;
            } else if (op.type == "ConvTranspose") {
                /* Todo */
            }
        }
    }

    _extractFunc(func, prefix="") {
        var result = new Map;
        for (let key in func.attr) {
            let value = func.attr[key];
            if (Object.prototype.hasOwnProperty.call(value, 'func')) {
                let resultNextLevel = this._extractFunc(value.func, prefix+key+".");
                for (let [key2, value2] of resultNextLevel) {
                    result.set(key2, value2);
                }
                continue;
            }
            result.set(key2, value2);
        }
        return result;
    }

    get name() {
        return this._name || '';
    }

    get type() {
        return this._type;
    }

    get metadata() {
        let schema = this ._metadata.type(this.type);
        return schema;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

    get chains() {
        return this._chains;
    }

    get device() {
        return this._device;
    }

    get controlDependencies() {
        return this._controlDependencies;
    }
};

om.Parameter = class {

    constructor(name, visible, args) {
        this._name = name;
        this._visible = visible;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

om.Argument = class {

    constructor(name, type, initializer) {
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get name() {
        return this._name;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

om.Attribute = class {

    constructor(type, name, value, schema, visible) {
        this._type = type;
        this._value = value;
        this._name = name;
        this._visible = visible;

        if (Object.prototype.hasOwnProperty.call(value, 'i')) {
            this._value = value.i;
        } else if (Object.prototype.hasOwnProperty.call(value, 'f')) {
            this._value = value.f;
        } else if (Object.prototype.hasOwnProperty.call(value, 'b')) {
            this._value = value.b;
        } else if (Object.prototype.hasOwnProperty.call(value, 'bt')) {
            if (0 != value.bt.length) {
                this._type = "tensor";
                this._value = new om.Tensor('Constant', new om.TensorType("float32", [value.bt.length/4], null), value.bt);
            } else {
                this._value = "NIL";
            }
        } else if (Object.prototype.hasOwnProperty.call(value, 's')) {
            if (typeof value.s === "string") {
                this._value = value.s;
            } else if (value.s.filter(c => c <= 32 && c >= 128).length == 0) {
                this._value = om.Metadata.textDecoder.decode(value.s);
            } else {
                this._value = value.s;
            }
        } else if (Object.prototype.hasOwnProperty.call(value, 'list')) {
            let list = value.list;
            this._value = [];
            if (list.s && list.s.length > 0) {
                this._value = list.i;
            } else if (list.i && list.i.length > 0) {
                this._value = list.i;
            } else if (list.f && list.f.length > 0) {
                this._value = list.f;
            } else if (list.type && list.type.length > 0) {
                this._type = 'type[]';
                this._value = list.type.map((type) => om.Tensor.formatDataType(type));
            } else if (list.shape && list.shape.length > 0) {
                this._type = 'shape[]';
                this._value = list.shape.map((shape) => new om.TensorShape(shape));
            }
        }

        let attrMeta = schema.attributes.find(item => item.name == name);
        if (attrMeta && attrMeta.type) {
            if (attrMeta.type == "Enum") {
                this._value = attrMeta.enum[this._value];
            }
            if (attrMeta.type == "DataType") {
                this._value = om.Node.enum2Dtype(this._value);
            }
            if (attrMeta.type == "DataTypeInner") {
                this._value = om.Node.enum2DtypeInner(this._value);
            }
            if (attrMeta.type == "Format") {
                const FORMAT_LIST = ["NCHW", "NHWC", "ND", "NC1HWC0", "Fractal-Z", "NC1C0HW Pad", "NHWC1C0", "FSR_NCHW",
                    "Fractal-Deconv", "C1HWNC0", "Fractal-Deconv-Transposed", "Fractal-Deconv-SP-Stride-Trans",
                    "NC1HWC0(C04)", "Fractal-Z(C04)", "CHWN", "Fractal-Deconc-SP-Stride8-Trans", "HWCN", "NC1KHKWHWC0",
                    "BN", "Filter HWCK"];
                this._value = FORMAT_LIST[this._value];
            }
            if (attrMeta.type == "Padding") {
                const PADDING_LIST = ["Ceil (Legacy)", "Direct Assign", "Valid (Legacy)", "Same (Legacy)", "Ceil",
                    "Valid", "Same"];
                this._value = PADDING_LIST[this._value];
            }
        }
    }

    toString() {
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible;
    }
};

om.Tensor = class {

    constructor(kind, type, value) {
        this._type = type;
        this._name = "";
        this._kind = kind;
        this._data = value;
        this._shape = type.shape.dimensions;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get kind() {
        return this._kind;
    }

    set kind(value) {
        this._kind = value;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        context.shape = this._type.rawShape.dimensions;
        return this._decode(context, 0);
    }

    _context() {
        let context = {};
        context.state = null;
        if (this._data == null) {
            context.state = "Tensor data is empty.";
            return context;
        }

        context.index = 0;
        context.count = 0;
        context.dataType = this._type.dataType;
        context.shape = this._shape;
        context.rawData = this._data;
        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        let results = [];
        let size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; ++i) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                if (context.data) {
                    results.push(this._decodeDataValue(context));
                    context.count++;
                } else {
                    if (context.rawData) {
                        const view = new DataView(context.rawData.buffer, context.rawData.byteOffset, context.rawData.length);
                        switch (context.dataType) {
                            case "float32":
                                var v = view.getFloat32(context.index, true);
                                results.push(v);
                                context.index += 4;
                                context.count++;
                                break;
                            case "int32":
                                var v = view.getInt32(context.index, true);
                                results.push(v);
                                context.index += 4;
                                context.count++;
                                break;
                            case "int64":
                                var v = view.getBigInt64(context.index, true);
                                results.push(Number(v));
                                context.index += 8;
                                context.count++;
                                break;
                            case "uint64":
                                var v = view.getBigUint64(context.index, true);
                                results.push(Number(v));
                                context.index += 8;
                                context.count++;
                                break;
                            case "uint32":
                                var v = view.getUint32(context.index, true);
                                results.push(v);
                                context.index += 4;
                                context.count++;
                                break;
                            case "float16":
                                var v = view.getFloat16(context.index, true);
                                results.push(v);
                                context.index += 2;
                                context.count++;
                                break;
                            case "int8":
                                var v = view.getInt8(context.index, true);
                                results.push(v);
                                context.index += 1;
                                context.count++;
                                break;
                            case "uint8":
                                var v = view.getUint8(context.index, true);
                                results.push(v);
                                context.index += 1;
                                context.count++;
                                break;
                        }
                    }
                }
            }
        } else {
            for (let j = 0; j < size; ++j) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    toString() {
        if (this,_type.shape != this._type.rawShape) {
            if (this._type.rawShape.unknown) {
                return 'Unresolved format, export as 1-D array';
            } else {
                return 'Visualization unavailable, export for checking out';
            }
        }

        let context = this._context();
        if (context.state) {
            return '';
        }

        if (context.rawData.length == 0) {
            return 'Internal Error';
        }
        context.limit = 50;
        let value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }
};

om.TensorType = class {

    constructor(dataType, shape, denotation) {
        this.dataType = dataType;
        this.shape = shape;
        this.denotation = denotation;
    }

    equals(obj) {
        return obj && this.dataType === obj.dataType && this.shape && this.shape.equals(obj.shape);
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

om.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions.map((dim) => Number.isInteger(dim) ? dim : dim.toNumber());
    }

    equals(obj) {
        if (obj && Array.isArray(obj.dimensions) && Array.isArray(this.dimensions)) {
            if (this.dimensions.length === obj.dimensions.length) {
                return obj.dimensions.every((value, index) => this.dimensions[index] === value);
            }
            if (obj.dimensions.every((dim) => Number.isInteger(dim)) && this.dimensions.every((dim) => Number.isInteger(dim))) {
                const a = obj.dimensions.reduce((a, b) => a * b, 1);
                const b = this.dimensions.reduce((a, b) => a * b, 1);
                return a === b;
            }
        }
        return false;
    }

    toString() {
        if (this.dimensions && Array.isArray(this.dimensions) && this.dimensions.length > 0) {
            return '[' + this.dimensions.map((dim) => dim ? dim.toString() : '?').join(',') + ']';
        }
        return '';
    }
};

om.Utility = class {

    static dtype(value) {
        om.Utility._types = om.Utility._types || [
            'undefined', 'float32', 'float16', 'int8', 'uint8', 'int16', 'uint16', 'int32',
            'int64', 'uint32', 'uint64', 'boolean', 'float64', 'string', 'dual_sub_int8', 'dual_sub_uint8',
            'complex64', 'complex128', 'qint8', 'qint16', 'qint32', 'quint8', 'quint16', 'resource',
            'stringref', 'dual', 'variant', 'bfloat16', 'int4', 'uint1', 'int2', 'uint2'
        ];
        if (value >= om.Utility._types.length) {
            throw new om.Error("Unsupported dtype '" + value + "'.");
        }
        return om.Utility._types[value];
    }

    static tensorType(desc) {
        if (desc.shape && Array.isArray(desc.shape.dim)) {
            const dataType = desc && desc.dtype ? om.Utility.dtype(desc.dtype) : '?';
            const shape = new om.TensorShape(desc.shape.dim);
            return new om.TensorType(dataType, shape, desc.layout);
        }
        return null;
    }

    static decodeText(value) {
        om.Utility._textDecoder = om.Utility._textDecoder || new TextDecoder('utf-8');
        return om.Utility._textDecoder.decode(value);
    }
};

om.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci model.';
    }
};

svp.ModelDef = class ModelDef {

    constructor(buffer) {
        const reader = new svp.BinaryReader(buffer);
        this.attr = {};
        this.graph = [];
        this.name = reader.find(0x800D, 'string');
        this.batch_num = reader.find(0x600A);
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            const value = reader.value(tag);
            switch (tag & 0x1fff) {
                case 0x0040: {
                    this.graph.push(new svp.GraphDef(value));
                    break;
                }
                case 0x0111: {
                    const op = new svp.OpDef(value);
                    for (const item of this.graph) {
                        if (op.attr && op.attr.seg_id && op.attr.seg_id.i === item.id) {
                            let out_num;
                            if (typeof op.output_index == 'number') {
                                out_num = op.output_index + 1;
                            } else {
                                const input_num = op.input.map((element) => element.split(":")[1]);
                                out_num = input_num.length > 0 ? Math.max(...input_num) + 1 : 1;
                            }
                            const out_types = [];
                            if (op.data_flow && op.data_flow !== '') {
                                const data = op.data_flow;
                                if (data.indexOf('o[{t') !== -1) {
                                    const outs = data.substring(data.indexOf('o[{t')).split(',');
                                    for (const out of outs) {
                                        const startIndex = out.indexOf("\"");
                                        const endIndex = out.indexOf("\"", startIndex + 1);
                                        out_types.push(out.substring(startIndex + 1, endIndex));
                                    }
                                }
                            }
                            const out_list = [];
                            while (out_num > 0) {
                                const output_desc = {};
                                output_desc.shape = { dim: op.output_shape_vector };
                                output_desc.layout = 'NCHW';
                                if (op.data_flow && out_types.length >= out_num) {
                                    output_desc.dtype = out_types[op.output_index + 1 - out_num];
                                }
                                out_list.push(output_desc);
                                out_num--;
                            }

                            let curr_op = null;
                            for (const op_item of item.op) {
                                if (op_item.id === op.id) {
                                    curr_op = op_item;
                                    break;
                                }
                            }
                            if (curr_op != null) {
                                curr_op.output_desc = curr_op.output_desc.concat(out_list);
                            } else {
                                op.output_desc = op.output_desc.concat(out_list);
                                item.op.push(op);
                            }
                            break;
                        }
                    }
                    break;
                }
                default: {
                    break;
                }
            }
        }
        if (this.graph.length > 1) {
            for (let i = 1; i < this.graph.length; i++) {
                this.graph[0].op = this.graph[0].op.concat(this.graph[i].op);
            }
        }
    }
};

svp.GraphDef = class {

    constructor(buffer) {
        this.input = [];
        this.output = [];
        this.op = [];
        this.attr = {};
        const reader = new svp.BinaryReader(buffer);
        const input = (buffer) => {
            const input = {};
            const reader = new svp.BinaryReader(buffer);
            while (reader.position < reader.length) {
                const tag = reader.uint16();
                switch (tag & 0x1fff) {
                    case 0x0051: input.id = reader.value(tag); break;
                    case 0x0058: input.name = reader.value(tag, 'string').trim(); break;
                    case 0x005a: input.shape_vector = reader.value(tag, 'uint32[]'); break;
                    default: reader.value(tag); break;
                }
            }
            return input;
        };
        const output = (buffer) => {
            const output = {};
            const reader = new svp.BinaryReader(buffer);
            while (reader.position < reader.length) {
                const tag = reader.uint16();
                switch (tag & 0x1fff) {
                    case 0x0061: output.id = reader.value(tag); break;
                    case 0x0066: output.name = reader.value(tag, 'string').trim(); break;
                    case 0x0069: output.shape_vector = reader.value(tag, 'uint32[]'); break;
                    case 0x0110: output.layer_num = reader.value(tag); break;
                    default: reader.value(tag); break;
                }
            }
            return output;
        };
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            const value = reader.value(tag);
            switch (tag & 0x1fff) {
                case 0x0041: this.id = value; break;
                case 0x0050: this.input.push(input(value)); break;
                case 0x0060: this.output.push(output(value)); break;
                default: break;
            }
        }
    }
};

svp.OpDef = class {

    constructor(buffer) {
        this.input = [];
        this.attr = {};
        this.input_i = [];
        this.output_i = [];
        this.input_desc = [];
        this.output_desc = [];
        const reader = new svp.BinaryReader(buffer);
        while (reader.position < reader.length) {
            const tag = reader.uint16();
            switch (tag & 0x1fff) {
                case 0x0114: this.name = reader.value(tag, 'string').trim(); break;
                case 0x0112: this.id = reader.value(tag); break;
                case 0x0119: this.attr.output_m2m_flag = reader.attribute(tag, 'i'); break;
                case 0x0121: this.attr.batch_flag = reader.attribute(tag, 'i'); break;
                case 0x0124: this.attr.dequant_scale = reader.attribute(tag, 'i'); break;
                case 0x0126: this.attr.output_address = reader.attribute(tag, 'i'); break;
                case 0x0125: this.attr.dequant_offset = reader.attribute(tag, 'i'); break;
                case 0x0127: this.attr.first_inst_addr = reader.attribute(tag, 'i'); break;
                case 0x0128: this.attr.last_inst_addr = reader.attribute(tag, 'i'); break;
                case 0x013B: this.attr.is_fusion_layer = reader.attribute(tag, 'i'); break;
                case 0x013C: this.input = reader.value(tag, 'string').split(','); break;
                case 0x014B: this.attr.seg_id = reader.attribute(tag, 'i'); break;
                case 0x0150: this.attr.is_not_last_merge_layer = reader.attribute(tag, 'i'); break;
                case 0x0151: this.attr.is_dump_avavilable = reader.attribute(tag, 'i'); break;
                case 0x0153: this.attr.debug_dump_offset = reader.attribute(tag, 'i'); break;
                case 0x0152: this.type = reader.value(tag, 'string'); break;
                case 0x0154: this.output_shape_vector = reader.value(tag, 'uint32[]'); break;
                case 0x0155: this.input_index = reader.value(tag); break;
                case 0x015B: this.output_index = reader.value(tag); break;
                case 0x0156: this.attr.trap_inst_pc = reader.attribute(tag, 'i'); break;
                case 0x0157: this.attr.profile_layer_id = reader.attribute(tag, 'i'); break;
                case 0xA15A:
                    this.data_flow = reader.value(tag, 'string');
                    this.attr.data_flow = new svp.AttrDef(this.data_flow.replace('i[{t', 'input[{type').replace(',f[{t', '\tforward[{type').replace(',o[{t', '\toutput[{type').replace(',{[t', ',{type'), 's');
                    break;
                default: reader.value(tag); break;
            }
        }
        for (let i = 0; i < this.input.length; i++) {
            this.input_desc.push({ layout: 'NCHW', shape: {} });
        }
    }
};

svp.AttrDef = class {

    constructor(item, type) {
        switch (type) {
            case 's': this.s = item; break;
            case 'i': this.i = item; break;
            default: throw new svp.Error("Unsupported attribute type '" + type + "'.");
        }
    }

    get value() {
        if (this.s !== undefined) {
            return 's';
        }
        if (this.i !== undefined) {
            return 'i';
        }
        return undefined;
    }
};

svp.BinaryReader = class extends base.BinaryReader {

    value(tag, type) {
        let value;
        switch (tag >> 13) {
            case 1: value = this.int8(); break;
            case 2: value = this.uint16(); break;
            case 3: value = this.uint32(); break;
            case 4: value = this.read(this.int8()); break;
            case 5: value = this.read(this.uint16()); break;
            case 6: value = this.read(this.uint32()); break;
            default: throw new svp.Error("Unsupported value identifier '" + tag + "'.");
        }
        return type ? this._cast(value, type, tag) : value;
    }

    find(tag, type) {
        let value = null;
        let match = false;
        while (!match && this.position < this.length) {
            const current = this.uint16();
            value = this.value(current);
            match = current === tag;
        }
        this.seek(0);
        return match && type ? this._cast(value, type, tag) : value;
    }

    attribute(tag, type) {
        const value = this.value(tag);
        return new svp.AttrDef(value, type);
    }

    _cast(value, type, tag) {
        switch (type) {
            case 'string': {
                if (value instanceof Uint8Array) {
                    svp.BinaryReader._decoder = svp.BinaryReader._decoder || new TextDecoder('utf-8');
                    return svp.BinaryReader._decoder.decode(value).replace(/\0.*$/g, '');
                }
                throw new om.Error("Invalid 'string' tag '" + tag.toString(16) + "'.");
            }
            case 'uint32[]': {
                const reader = new base.BinaryReader(value);
                value = [];
                while (reader.position < reader.length) {
                    value.push(reader.uint32());
                }
                return value;
            }
            default: {
                return value;
            }
        }
    }
};

svp.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DaVinci SVP model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = om.ModelFactory;
}
