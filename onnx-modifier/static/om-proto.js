var $root = protobuf.get('om');

$root.ge = {};

$root.om = {};

$root.om.DataType = {
    "DT_UNDEFINED": 0,
    "DT_FLOAT": 1,
    "DT_FLOAT16": 2,
    "DT_INT8": 3,
    "DT_UINT8": 4,
    "DT_INT16": 5,
    "DT_UINT16": 6,
    "DT_INT32": 7,
    "DT_INT64": 8,
    "DT_UINT32": 9,
    "DT_UINT64": 10,
    "DT_BOOL": 11,
    "DT_DOUBLE": 12,
    "DT_STRING": 13,
    "DT_DUAL_SUB_INT8": 14,
    "DT_DUAL_SUB_UINT8": 15,
    "DT_COMPLEX64": 16,
    "DT_COMPLEX128": 17,
    "DT_QINT8": 18,
    "DT_QINT16": 19,
    "DT_QINT32": 20,
    "DT_QUINT8": 21,
    "DT_QUINT16": 22,
    "DT_RESOURCE": 23,
    "DT_STRING_REF": 24,
    "DT_DUAL": 25,
    "DT_VARIANT": 26,
    "DT_BF16": 27,
    "DT_INT4": 28,
    "DT_UINT1": 29,
    "DT_INT2": 30,
    "DT_UINT2": 31
};

$root.om.AttrDef = class AttrDef {

    constructor() {
        AttrDef.prototype.index = 0
    }

    get value() {
        $root.om.AttrDef.valueSet = $root.om.AttrDef.valueSet || new Set([ "s", "i", "f", "b", "bt", "list", "func", "td", "t", "g", "list_list_int", "dt", "list_list_float"]);
        return Object.keys(this).find((key) => $root.om.AttrDef.valueSet.has(key) && this[key] != null);
    }

    static decode(reader, length) {
        const message = new $root.om.AttrDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.s = reader.bytes();
                    break;
                case 3:
                    message.i = reader.int64();
                    break;
                case 4:
                    message.f = reader.float();
                    break;
                case 5:
                    message.b = reader.bool();
                    break;
                case 7:
                    message.bt = reader.bytes();
                    break;
                case 1:
                    message.list = $root.om.AttrDef.ListValue.decode(reader, reader.uint32());
                    break;
                case 10:
                    message.func = $root.om.NamedAttrs.decode(reader, reader.uint32());
                    break;
                case 11:
                    message.td = $root.om.TensorDescriptor.decode(reader, reader.uint32());
                    break;
                case 12:
                    message.t = $root.om.TensorDef.decode(reader, reader.uint32());
                    break;
                case 13:
                    message.g = $root.om.GraphDef.decode(reader, reader.uint32());
                    break;
                case 14:
                    message.list_list_int = $root.om.AttrDef.ListListInt.decode(reader, reader.uint32());
                    break;
                case 15:
                    message.dt = reader.int64();
                    break;
                case 16:
                    message.list_list_float = $root.om.AttrDef.ListListFloat.decode(reader, reader.uint32());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.AttrDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    message.s = reader.bytes();
                    break;
                case "i":
                    message.i = reader.int64();
                    break;
                case "f":
                    message.f = reader.float();
                    break;
                case "b":
                    message.b = reader.bool();
                    break;
                case "bt":
                    message.bt = reader.bytes();
                    break;
                case "list":
                    message.list = $root.om.AttrDef.ListValue.decodeText(reader);
                    break;
                case "func":
                    message.func = $root.om.NamedAttrs.decodeText(reader);
                    break;
                case "td":
                    message.td = $root.om.TensorDescriptor.decodeText(reader);
                    break;
                case "t":
                    message.t = $root.om.TensorDef.decodeText(reader);
                    break;
                case "g":
                    message.g = $root.om.GraphDef.decodeText(reader);
                    break;
                case "list_list_int":
                    message.list_list_int = $root.om.AttrDef.ListListInt.decodeText(reader);
                    break;
                case "dt":
                    message.dt = reader.int64();
                    break;
                case "list_list_float":
                    message.list_list_float = $root.om.AttrDef.ListListFloat.decodeText(reader);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.AttrDef.ListValue = class ListValue {

    constructor() {
        this.s = [];
        this.i = [];
        this.f = [];
        this.b = [];
        this.bt = [];
        this.td = [];
        this.t = [];
        this.g = [];
        this.na = [];
        this.dt = [];
    }

    static decode(reader, length) {
        const message = new $root.om.AttrDef.ListValue();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 2:
                    message.s.push(reader.bytes());
                    break;
                case 3:
                    message.i = reader.array(message.i, () => reader.int64(), tag);
                    break;
                case 4:
                    message.f = reader.floats(message.f, tag);
                    break;
                case 5:
                    message.b = reader.array(message.b, () => reader.bool(), tag);
                    break;
                case 7:
                    message.bt.push(reader.bytes());
                    break;
                case 8:
                    message.td.push($root.om.TensorDescriptor.decode(reader, reader.uint32()));
                    break;
                case 9:
                    message.t.push($root.om.TensorDef.decode(reader, reader.uint32()));
                    break;
                case 10:
                    message.g.push($root.om.GraphDef.decode(reader, reader.uint32()));
                    break;
                case 11:
                    message.na.push($root.om.NamedAttrs.decode(reader, reader.uint32()));
                    break;
                case 12:
                    message.dt = reader.array(message.dt, () => reader.int64(), tag);
                    break;
                case 20:
                    message.val_type = reader.int32();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.AttrDef.ListValue();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "s":
                    reader.array(message.s, () => reader.bytes());
                    break;
                case "i":
                    reader.array(message.i, () => reader.int64());
                    break;
                case "f":
                    reader.array(message.f, () => reader.float());
                    break;
                case "b":
                    reader.array(message.b, () => reader.bool());
                    break;
                case "bt":
                    reader.array(message.bt, () => reader.bytes());
                    break;
                case "td":
                    message.td.push($root.om.TensorDescriptor.decodeText(reader));
                    break;
                case "t":
                    message.t.push($root.om.TensorDef.decodeText(reader));
                    break;
                case "g":
                    message.g.push($root.om.GraphDef.decodeText(reader));
                    break;
                case "na":
                    message.na.push($root.om.NamedAttrs.decodeText(reader));
                    break;
                case "dt":
                    reader.array(message.dt, () => reader.int64());
                    break;
                case "val_type":
                    message.val_type = reader.enum($root.om.AttrDef.ListValue.ListValueType);
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.AttrDef.ListValue.prototype.val_type = 0;

$root.om.AttrDef.ListValue.ListValueType = {
    "VT_LIST_NONE": 0,
    "VT_LIST_STRING": 1,
    "VT_LIST_INT": 2,
    "VT_LIST_FLOAT": 3,
    "VT_LIST_BOOL": 4,
    "VT_LIST_BYTES": 5,
    "VT_LIST_TENSOR_DESC": 6,
    "VT_LIST_TENSOR": 7,
    "VT_LIST_GRAPH": 8,
    "VT_LIST_NAMED_ATTRS": 9,
    "VT_LIST_DATA_TYPE": 10
};

$root.om.AttrDef.ListListInt = class ListListInt {

    constructor() {
        this.list_list_i = [];
    }

    static decode(reader, length) {
        const message = new $root.om.AttrDef.ListListInt();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.list_list_i.push($root.om.AttrDef.ListListInt.ListInt.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.AttrDef.ListListInt();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "list_list_i":
                    message.list_list_i.push($root.om.AttrDef.ListListInt.ListInt.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.AttrDef.ListListInt.ListInt = class ListInt {

    constructor() {
        this.list_i = [];
    }

    static decode(reader, length) {
        const message = new $root.om.AttrDef.ListListInt.ListInt();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.list_i = reader.array(message.list_i, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.AttrDef.ListListInt.ListInt();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "list_i":
                    reader.array(message.list_i, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.AttrDef.ListListFloat = class ListListFloat {

    constructor() {
        this.list_list_f = [];
    }

    static decode(reader, length) {
        const message = new $root.om.AttrDef.ListListFloat();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.list_list_f.push($root.om.AttrDef.ListListFloat.ListFloat.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.AttrDef.ListListFloat();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "list_list_f":
                    message.list_list_f.push($root.om.AttrDef.ListListFloat.ListFloat.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.AttrDef.ListListFloat.ListFloat = class ListFloat {

    constructor() {
        this.list_f = [];
    }

    static decode(reader, length) {
        const message = new $root.om.AttrDef.ListListFloat.ListFloat();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.list_f = reader.floats(message.list_f, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.AttrDef.ListListFloat.ListFloat();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "list_f":
                    reader.array(message.list_f, () => reader.float());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.NamedAttrs = class NamedAttrs {

    constructor() {
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.om.NamedAttrs();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.NamedAttrs();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.NamedAttrs.prototype.name = "";

$root.om.ShapeDef = class ShapeDef {

    constructor() {
        this.dim = [];
    }

    static decode(reader, length) {
        const message = new $root.om.ShapeDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.dim = reader.array(message.dim, () => reader.int64(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.ShapeDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "dim":
                    reader.array(message.dim, () => reader.int64());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.TensorDescriptor = class TensorDescriptor {

    constructor() {
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.om.TensorDescriptor();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.dtype = reader.int32();
                    break;
                case 3:
                    message.shape = $root.om.ShapeDef.decode(reader, reader.uint32());
                    break;
                case 4:
                    message.layout = reader.string();
                    break;
                case 9:
                    message.has_out_attr = reader.bool();
                    break;
                case 10:
                    message.size = reader.int64();
                    break;
                case 11:
                    message.weight_size = reader.int64();
                    break;
                case 12:
                    message.reuse_input = reader.bool();
                    break;
                case 13:
                    message.output_tensor = reader.bool();
                    break;
                case 14:
                    message.device_type = reader.string();
                    break;
                case 15:
                    message.input_tensor = reader.bool();
                    break;
                case 16:
                    message.real_dim_cnt = reader.int64();
                    break;
                case 17:
                    message.reuse_input_index = reader.int64();
                    break;
                case 18:
                    message.data_offset = reader.int64();
                    break;
                case 19:
                    message.cmps_size = reader.int64();
                    break;
                case 20:
                    message.cmps_tab = reader.string();
                    break;
                case 21:
                    message.cmps_tab_offset = reader.int64();
                    break;
                case 5:
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.TensorDescriptor();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "dtype":
                    message.dtype = reader.enum($root.om.DataType);
                    break;
                case "shape":
                    message.shape = $root.om.ShapeDef.decodeText(reader);
                    break;
                case "layout":
                    message.layout = reader.string();
                    break;
                case "has_out_attr":
                    message.has_out_attr = reader.bool();
                    break;
                case "size":
                    message.size = reader.int64();
                    break;
                case "weight_size":
                    message.weight_size = reader.int64();
                    break;
                case "reuse_input":
                    message.reuse_input = reader.bool();
                    break;
                case "output_tensor":
                    message.output_tensor = reader.bool();
                    break;
                case "device_type":
                    message.device_type = reader.string();
                    break;
                case "input_tensor":
                    message.input_tensor = reader.bool();
                    break;
                case "real_dim_cnt":
                    message.real_dim_cnt = reader.int64();
                    break;
                case "reuse_input_index":
                    message.reuse_input_index = reader.int64();
                    break;
                case "data_offset":
                    message.data_offset = reader.int64();
                    break;
                case "cmps_size":
                    message.cmps_size = reader.int64();
                    break;
                case "cmps_tab":
                    message.cmps_tab = reader.string();
                    break;
                case "cmps_tab_offset":
                    message.cmps_tab_offset = reader.int64();
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.TensorDescriptor.prototype.name = "";
$root.om.TensorDescriptor.prototype.dtype = 0;
$root.om.TensorDescriptor.prototype.shape = null;
$root.om.TensorDescriptor.prototype.layout = "";
$root.om.TensorDescriptor.prototype.has_out_attr = false;
$root.om.TensorDescriptor.prototype.size = protobuf.Int64.create(0);
$root.om.TensorDescriptor.prototype.weight_size = protobuf.Int64.create(0);
$root.om.TensorDescriptor.prototype.reuse_input = false;
$root.om.TensorDescriptor.prototype.output_tensor = false;
$root.om.TensorDescriptor.prototype.device_type = "";
$root.om.TensorDescriptor.prototype.input_tensor = false;
$root.om.TensorDescriptor.prototype.real_dim_cnt = protobuf.Int64.create(0);
$root.om.TensorDescriptor.prototype.reuse_input_index = protobuf.Int64.create(0);
$root.om.TensorDescriptor.prototype.data_offset = protobuf.Int64.create(0);
$root.om.TensorDescriptor.prototype.cmps_size = protobuf.Int64.create(0);
$root.om.TensorDescriptor.prototype.cmps_tab = "";
$root.om.TensorDescriptor.prototype.cmps_tab_offset = protobuf.Int64.create(0);

$root.om.TensorDef = class TensorDef {

    constructor() {
    }

    static decode(reader, length) {
        const message = new $root.om.TensorDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.desc = $root.om.TensorDescriptor.decode(reader, reader.uint32());
                    break;
                case 2:
                    message.data = reader.bytes();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.TensorDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "desc":
                    message.desc = $root.om.TensorDescriptor.decodeText(reader);
                    break;
                case "data":
                    message.data = reader.bytes();
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.TensorDef.prototype.desc = null;
$root.om.TensorDef.prototype.data = new Uint8Array([]);

$root.om.OpDef = class OpDef {

    constructor() {
        this.input = [];
        this.attr = {};
        this.input_name = [];
        this.src_name = [];
        this.src_index = [];
        this.dst_name = [];
        this.dst_index = [];
        this.input_i = [];
        this.output_i = [];
        this.workspace = [];
        this.workspace_bytes = [];
        this.is_input_const = [];
        this.input_desc = [];
        this.output_desc = [];
        this.subgraph_name = [];
    }

    static decode(reader, length) {
        const message = new $root.om.OpDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.string();
                    break;
                case 5:
                    message.input.push(reader.string());
                    break;
                case 10:
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decode(reader, reader.uint32()));
                    break;
                case 20:
                    message.has_out_attr = reader.bool();
                    break;
                case 21:
                    message.id = reader.int64();
                    break;
                case 22:
                    message.stream_id = reader.int64();
                    break;
                case 23:
                    message.input_name.push(reader.string());
                    break;
                case 24:
                    message.src_name.push(reader.string());
                    break;
                case 25:
                    message.src_index = reader.array(message.src_index, () => reader.int64(), tag);
                    break;
                case 26:
                    message.dst_name.push(reader.string());
                    break;
                case 27:
                    message.dst_index = reader.array(message.dst_index, () => reader.int64(), tag);
                    break;
                case 28:
                    message.input_i = reader.array(message.input_i, () => reader.int64(), tag);
                    break;
                case 29:
                    message.output_i = reader.array(message.output_i, () => reader.int64(), tag);
                    break;
                case 30:
                    message.workspace = reader.array(message.workspace, () => reader.int64(), tag);
                    break;
                case 31:
                    message.workspace_bytes = reader.array(message.workspace_bytes, () => reader.int64(), tag);
                    break;
                case 32:
                    message.is_input_const = reader.array(message.is_input_const, () => reader.bool(), tag);
                    break;
                case 33:
                    message.input_desc.push($root.om.TensorDescriptor.decode(reader, reader.uint32()));
                    break;
                case 34:
                    message.output_desc.push($root.om.TensorDescriptor.decode(reader, reader.uint32()));
                    break;
                case 35:
                    message.subgraph_name.push(reader.string());
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.OpDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "type":
                    message.type = reader.string();
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decodeText(reader));
                    break;
                case "has_out_attr":
                    message.has_out_attr = reader.bool();
                    break;
                case "id":
                    message.id = reader.int64();
                    break;
                case "stream_id":
                    message.stream_id = reader.int64();
                    break;
                case "input_name":
                    reader.array(message.input_name, () => reader.string());
                    break;
                case "src_name":
                    reader.array(message.src_name, () => reader.string());
                    break;
                case "src_index":
                    reader.array(message.src_index, () => reader.int64());
                    break;
                case "dst_name":
                    reader.array(message.dst_name, () => reader.string());
                    break;
                case "dst_index":
                    reader.array(message.dst_index, () => reader.int64());
                    break;
                case "input_i":
                    reader.array(message.input_i, () => reader.int64());
                    break;
                case "output_i":
                    reader.array(message.output_i, () => reader.int64());
                    break;
                case "workspace":
                    reader.array(message.workspace, () => reader.int64());
                    break;
                case "workspace_bytes":
                    reader.array(message.workspace_bytes, () => reader.int64());
                    break;
                case "is_input_const":
                    reader.array(message.is_input_const, () => reader.bool());
                    break;
                case "input_desc":
                    message.input_desc.push($root.om.TensorDescriptor.decodeText(reader));
                    break;
                case "output_desc":
                    message.output_desc.push($root.om.TensorDescriptor.decodeText(reader));
                    break;
                case "subgraph_name":
                    reader.array(message.subgraph_name, () => reader.string());
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.OpDef.prototype.name = "";
$root.om.OpDef.prototype.type = "";
$root.om.OpDef.prototype.has_out_attr = false;
$root.om.OpDef.prototype.id = protobuf.Int64.create(0);
$root.om.OpDef.prototype.stream_id = protobuf.Int64.create(0);

$root.om.GraphDef = class GraphDef {

    constructor() {
        this.input = [];
        this.output = [];
        this.op = [];
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.om.GraphDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 4:
                    message.input.push(reader.string());
                    break;
                case 5:
                    message.output.push(reader.string());
                    break;
                case 6:
                    message.op.push($root.om.OpDef.decode(reader, reader.uint32()));
                    break;
                case 11:
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.GraphDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "input":
                    reader.array(message.input, () => reader.string());
                    break;
                case "output":
                    reader.array(message.output, () => reader.string());
                    break;
                case "op":
                    message.op.push($root.om.OpDef.decodeText(reader));
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.GraphDef.prototype.name = "";

$root.om.ModelDef = class ModelDef {

    constructor() {
        this.graph = [];
        this.attr = {};
    }

    static decode(reader, length) {
        const message = new $root.om.ModelDef();
        const end = length !== undefined ? reader.position + length : reader.length;
        while (reader.position < end) {
            const tag = reader.uint32();
            switch (tag >>> 3) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.version = reader.uint32();
                    break;
                case 3:
                    message.custom_version = reader.string();
                    break;
                case 7:
                    message.graph.push($root.om.GraphDef.decode(reader, reader.uint32()));
                    break;
                case 11:
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decode(reader, reader.uint32()));
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }

    static decodeText(reader) {
        const message = new $root.om.ModelDef();
        reader.start();
        while (!reader.end()) {
            const tag = reader.tag();
            switch (tag) {
                case "name":
                    message.name = reader.string();
                    break;
                case "version":
                    message.version = reader.uint32();
                    break;
                case "custom_version":
                    message.custom_version = reader.string();
                    break;
                case "graph":
                    message.graph.push($root.om.GraphDef.decodeText(reader));
                    break;
                case "attr":
                    reader.entry(message.attr, () => reader.string(), () => $root.om.AttrDef.decodeText(reader));
                    break;
                default:
                    reader.field(tag, message);
                    break;
            }
        }
        return message;
    }
};

$root.om.ModelDef.prototype.name = "";
$root.om.ModelDef.prototype.version = 0;
$root.om.ModelDef.prototype.custom_version = "";