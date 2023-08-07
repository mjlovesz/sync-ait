
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

function inNodeConst(op) {
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

    _loadPICO(context) {
        const stream = context.stream;
        const reader = new base.BinaryReader(stream);
        var buffer = reader.read(4);
        this.format = 'DaVinci OM SVP'; // SVP = Smart Vision PICO
        reader.uint32(); // reserved
        this.size = reader.uint32();
        const param_size = reader.uint32();
        const param_offset = reader.uint32();
        reader.uint32(); // tmp_bufsize
        const tfm_offset = reader.uint32();
        reader.uint32(); // tfm_size
        this.type = 2;
        reader.seek(param_offset);
        this.param = reader.read(param_size);
        buffer = reader.read(tfm_offset - reader.position);
        this.model = new svp.ModelDef(buffer);
    }
};

om.Metadata = class {
    static open(context) {
        om.Metadata._metadata = om.Metadata._metadata || new Map();
        return context.request('./om-metadata.json', 'utf-8', null).then((data) => {
            om.Metadata._metadata = new om.Metadata(data);
            return om.Metadata._metadata;
        }).catch(() => {
            om.Metadata._metadata = new om.Metadata(null);
            return om.Metadata._metadata;
        });
    }

    constructor(data) {
        if (data) {
            const metadata = JSON.parse(data);
            this._types = new Map();
            this._attributes = new Map();
            this._inputs = new Map();
            for (const item of metadata || []) {
                this._types.set(item.name, item);
                if (item.identifier !== undefined) {
                    this._types.set(item.identifier, item);
                }
            }
        }
    }

    type(name) {
        if (!this._types.has(name)) {
            this._types.set(name, { name: name.toString() });
        }
        return this._types.get(name);
    }
};

om.Model = class {

    constructor(metadata, target) {
        this.graphs = [];
        this.format = target.format;
        const context = { metadata: metadata, weights: target.weights };
        for (const graph of target.model.graph) {
            this.graphs.push(new om.Graph(context, graph));
        }
    }
};

om.Graph = class {

    constructor(context, graph) {
        this.name = graph.name;
        this.nodes = [];
        this.inputs = [];
        this.outputs = [];
        const values = new Map();
        const value = (name, type, tensor) => {
            if (!values.has(name)) {
                values.set(name, new om.Value(name, type || null, tensor || null));
            } else if ((type && !type.equals(values.get(name).type)) ||
                       (tensor && tensor !== values.get(name).initializer)) {
                throw new om.Error("Duplicate value '" + name + "'.");
            }
            return values.get(name);
        };
        const tensors = new Map();
        const ops = [];
        for (const op of graph.op) {
            if (op.type === 'Const' && op.attr && op.attr.value) {
                const desc = op.attr.value.t.desc;
                let data = null;
                if (op.attr.value.t.data.length !== 0) {
                    data = op.attr.value.t.data;
                } else if (context.weights == null) {
                    data = null;
                } else if (desc.attr.merged_offset) {
                    const offset = desc.attr.merged_offset.i;
                    data = context.weights.slice(offset, offset + desc.weight_size);
                } else {
                    const offset = desc.data_offset;
                    data = context.weights.slice(offset, offset + desc.weight_size);
                }
                const type = om.Utility.tensorType(desc);
                const tensor = new om.Tensor('Constant', type, data);
                tensors.set(op.name, tensor);
                continue;
            }
            ops.push(op);
        }
        for (const op of ops) {
            const node = new om.Node(context, op, graph, value, tensors);
            this.nodes.push(node);
        }
    }
};

om.Node = class {

    constructor(context, op, graph, value, tensors) {
        this.name = op.name || '';
        this.type = context.metadata.type(op.type) || { name: op.type };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        this.chain = [];
        this.controlDependencies = [];
        this.device = null;
        if (op.input) {
            for (let i = 0; i < op.input.length; i++) {
                const input = op.input[i];
                if (input === '') {
                    continue;
                }
                var name = this.type.inputs && i < this.type.inputs.length ? this.type.inputs[i].name : 'input' + (i === 0 ? '' : i.toString());
                name = (name == 'weights') ? 'W' : name;
                name = (name == 'bias') ? 'B' : name;
                const index = input.lastIndexOf(':');
                const identifier = input.substring(0, index);
                const src_index = input.substring(index + 1);
                if (src_index === '-1') {
                    this.controlDependencies.push(value(name));
                    continue;
                }
                const type = om.Utility.tensorType(op.input_desc[i]);
                const tensor = tensors.get(identifier);
                const argument = new om.Argument(name, [ value(input, type, tensor) ]);
                this.inputs.push(argument);
            }
        }
        if (op.output_desc) {
            for (let i = 0; i < op.output_desc.length; i++) {
                const identifier = this.name + ':' + i.toString();
                const type = om.Utility.tensorType(op.output_desc[i]);
                const name = this.type.outputs && i < this.type.outputs.length ? this.type.outputs[i].name : 'output' + (i === 0 ? '' : i.toString());
                const argument = new om.Argument(name, [ value(identifier, type) ]);
                this.outputs.push(argument);
            }
        }
        for (const attr of Object.entries(op.attr || {})) {
            const name = attr[0];
            const value = attr[1];
            if (name === 'device') {
                this.device = value;
                continue;
            }
            if (name === 'original_op_names') {
                continue;
            }
            if (name === 'relu_flag' && value.b) {
                this.chain.push(new om.Node(context, { type: 'ReLU' }, graph, value));
                continue;
            }
            const attribute = new om.Attribute(context, name, value);
            this.attributes.push(attribute);
        }
    }
};

om.Attribute = class {

    constructor(context, name, value) {
        this.name = name;
        this.value = value;
        switch (value.value) {
            case 'i': {
                this.value = value.i;
                this.type = 'int64';
                break;
            }
            case 'f': {
                this.value = value.f;
                this.type = 'float32';
                break;
            }
            case 'b': {
                this.value = value.b;
                this.type = 'boolean';
                break;
            }
            case 'bt': {
                this.value = null;
                if (value.bt.length !== 0) {
                    this.type = 'tensor';
                    const shape = new om.TensorShape([ value.bt.length / 4 ]);
                    const type = new om.TensorType('float32', shape);
                    this.value = new om.Tensor('Constant', type, value.bt);
                }
                break;
            }
            case 'dt': {
                this.type = 'DataType';
                this.value = om.Utility.dtype(value.dt.toNumber());
                break;
            }
            case 's': {
                if (typeof value.s === 'string') {
                    this.value = value.s;
                } else if (value.s.filter(c => c <= 32 && c >= 128).length === 0) {
                    this.value = om.Utility.decodeText(value.s);
                } else {
                    this.value = value.s;
                }
                this.type = 'string';
                break;
            }
            case 'g': {
                this.type = 'graph';
                this.value = new om.Graph(context, value.g);
                break;
            }
            case 'func': {
                break;
            }
            case 'list': {
                const list = value.list;
                this.value = [];
                if (list.s && list.s.length > 0) {
                    this.value = list.s.map(v => String.fromCharCode.apply(null, new Uint16Array(v))).join(', ');
                    this.type = 'string[]';
                } else if (list.b && list.b.length > 0) {
                    this.value = list.b;
                    this.type = 'boolean[]';
                } else if (list.i && list.i.length > 0) {
                    this.value = list.i;
                    this.type = 'int64[]';
                } else if (list.f && list.f.length > 0) {
                    this.value = list.f;
                    this.type = 'float32[]';
                } else if (list.type && list.type.length > 0) {
                    this.type = 'type[]';
                    this.value = list.type.map((type) => om.Node.enum2Dtype(type) || '?');
                } else if (list.shape && list.shape.length > 0) {
                    this.type = 'shape[]';
                    this.value = list.shape.map((shape) => new om.TensorShape(shape));
                }
                break;
            }
            case 'list_list_int': {
                this.value = value.list_list_int.list_list_i.map((list) => list.list_i);
                break;
            }
            case 't': {
                const type = om.Utility.tensorType(value.t.desc);
                this.value = new om.Tensor('Constant', type, value.t.bytes);
                this.type = 'tensor';
                break;
            }
            case undefined: {
                this.value = null;
                break;
            }
            default: {
                throw new om.Error("Unsupported attribute type '" + JSON.stringify(value).substring(0, 32) + "'.");
            }
        }
    }
};

om.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.arguments = value;
    }

    get visible() {
        return true;
    }
};

om.Value = class {

    constructor(name, type, initializer) {
        if (typeof name !== 'string') {
            throw new om.Error("Invalid value identifier '" + JSON.stringify(name) + "'.");
        }
        this.name = name;
        this._type = type || null;
        this.initializer = initializer || null;
    }

    get type() {
        if (this.initializer) {
            return this.initializer.type;
        }
        return this._type;
    }
};

om.Tensor = class {

    constructor(category, type, value) {
        this.category = category;
        this.type = type;
        this.data = value;
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
