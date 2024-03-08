import os
from llm.transform.utils import (
    check_libclang_so,
    filter_chinese_char,
    get_args_and_options,
    print_spelling,
    print_update_info,
    update_contents,
)
from llm.common.log import logger

USING_SCALE_BIAS_ITEMS = ["IN_QKV", "IN_QMIXED", "IN_KMIXED", "IN_VMIXED", "IN_SELFOUTLINEAR", "IN_MLP"]
INTERMIDATE_PREFIX = "INTERMIDATE_"
CPP_TEMP_FILE_NAME = "transform_quant_cpp_temp.cpp"
HPP_TEMP_FILE_NAME = "transform_quant_cpp_temp.hpp"
DEQSCALE_SUFFIX = "DEQSCALE"
BIAS_SUFFIX = "BIAS"
WEIGHT_SUFFIX = "WEIGHT"
IN_BETA = "IN_BETA"
IN_HOLDER = "IN_HOLDER"


def add_scale_bias_in_enum(contents, enum_cursor, indent=4):
    added_items, insert_position, is_intermodate_found = [], enum_cursor.extent.end.offset - 1, False
    all_enums, op_weight_with_bias = [], {}
    for enum_item in enum_cursor.get_children():
        enum_item_spelling = enum_item.spelling
        all_enums.append(enum_item_spelling)
        if not enum_item_spelling.endswith(BIAS_SUFFIX):
            continue
        enum_op_name = enum_item_spelling[:-len(BIAS_SUFFIX)]
        enum_op_name = enum_op_name[:-1] if enum_op_name.endswith("_") else enum_op_name
        if enum_op_name + WEIGHT_SUFFIX in all_enums:
            op_weight_with_bias[enum_op_name + WEIGHT_SUFFIX] = enum_item_spelling
        elif enum_op_name + "_" + WEIGHT_SUFFIX in all_enums:
            op_weight_with_bias[enum_op_name + "_" + WEIGHT_SUFFIX] = enum_item_spelling

    for enum_item in enum_cursor.get_children():
        enum_item_spelling = enum_item.spelling
        if any([ii in enum_item_spelling for ii in USING_SCALE_BIAS_ITEMS]):
            added_items.append(enum_item_spelling + "_" + DEQSCALE_SUFFIX)
            if enum_item_spelling not in op_weight_with_bias:
                added_items.append(enum_item_spelling + "_" + BIAS_SUFFIX)
        if not is_intermodate_found and enum_item_spelling.startswith(INTERMIDATE_PREFIX):
            insert_position = contents[: enum_item.extent.start.offset].rfind("\n") + 1
            is_intermodate_found = True

    if IN_BETA not in all_enums:
        added_items.append(IN_BETA)

    indent_prefix = " " * indent
    insert_contents = "".join([indent_prefix + ii + ",\n" for ii in added_items])
    insert_contents = "\n" + indent_prefix + "// Quant weights\n" + insert_contents + "\n"
    return (insert_contents, insert_position, insert_position), added_items


def update_in_tensor_count(contents, cursor, in_tensor_count_added):
    cur_contents = contents[cursor.extent.end.offset :]
    next_semicolon_pos = cur_contents.find(";")
    cur_in_tensor_count = int(cur_contents[:next_semicolon_pos].split("=")[-1].strip())
    cur_in_tensor_count += in_tensor_count_added
    insert_contents = " = {}".format(cur_in_tensor_count)
    insert_start = cursor.extent.end.offset
    insert_end = insert_start + next_semicolon_pos
    return insert_contents, insert_start, insert_end


def update_from_json(contents, cursor, in_tensor_added):
    json_param, in_param = list(cursor.get_arguments())[:2]
    json_param_spelling, in_param_spelling = json_param.spelling, in_param.spelling
    insert_position = cursor.extent.end.offset - 1
    param_format = "    " + json_param_spelling + '.at("{}").get_to(' + in_param_spelling + ".{});"
    insert_contents = "\n".join([param_format.format(ii.lower(), ii.lower()) for ii in in_tensor_added]) + "\n"
    return insert_contents, insert_position, insert_position


def is_layer_function(cursor):
    # This check may change depending on actual situation
    if len(list(cursor.get_arguments())) < 2:
        return False
    op_parameter = list(cursor.get_arguments())[1]
    parameter_type = "".join([ii.spelling for ii in op_parameter.get_tokens()][:3])
    return parameter_type == "atb::Operation"


def is_in_tensor_count(cur_spelling):
    return cur_spelling.startswith("IN_") and cur_spelling.endswith("_COUNT")


def parse_file_as_cursor(file_path):
    from clang import cindex

    file_ext = os.path.splitext(file_path)[-1]
    temp_file = CPP_TEMP_FILE_NAME if file_ext in [".c", ".cpp"] else HPP_TEMP_FILE_NAME

    contents = open(file_path).read()
    contents = filter_chinese_char(contents)
    args, options = get_args_and_options()
    parser = cindex.Index.create(excludeDecls=True)
    tu = parser.parse(temp_file, args=args, unsaved_files=[(temp_file, contents)])
    return tu.cursor, contents


def transform_quant_cpp(cpp_file_path, indent=4):
    from clang import cindex
    from llm.transform.transform_quant_cpp_layer_function import TransformQuantCppLayerFunction

    cursor, contents = parse_file_as_cursor(cpp_file_path)
    children = list(next(list(cursor.get_children())[-1].get_children()).get_children())
    print_spelling(children, info="Children parts from cpp: ", level="info")

    updates, in_tensor_added = [], []
    for cur_cursor in children:
        cur_spelling = cur_cursor.spelling
        print_spelling(cur_cursor, info=f"current cursor: {cur_spelling}, {cur_cursor.kind}, ")
        if cur_cursor.kind == cindex.CursorKind.ENUM_DECL:
            (insert_contents, insert_start, insert_end), in_tensor_added = add_scale_bias_in_enum(
                contents, cur_cursor, indent
            )
        elif cur_cursor.kind == cindex.CursorKind.VAR_DECL and is_in_tensor_count(cur_spelling):
            insert_contents, insert_start, insert_end = update_in_tensor_count(
                contents, cur_cursor, len(in_tensor_added)
            )
        elif cur_cursor.kind == cindex.CursorKind.FUNCTION_DECL and cur_spelling == "from_json":
            insert_contents, insert_start, insert_end = update_from_json(contents, cur_cursor, in_tensor_added)
        elif cur_cursor.kind == cindex.CursorKind.FUNCTION_DECL and is_layer_function(cur_cursor):
            cur_updates = TransformQuantCppLayerFunction(contents, cur_cursor, in_tensor_added, indent=indent)()
            updates.extend(cur_updates)
            continue
        else:
            continue
        updates.append((insert_start, insert_end, insert_contents))
        print_update_info(insert_contents, insert_start, insert_end)
    return update_contents(contents, updates), in_tensor_added


def transform_quant_hpp(h_file_path, in_tensor_added, indent=4):
    from clang import cindex

    cursor, contents = parse_file_as_cursor(h_file_path)
    children = list(next(list(cursor.get_children())[-1].get_children()).get_children())
    print_spelling(children, info="Children parts from cpp: ", level="info")

    indent_prefix = " " * indent
    updates = []
    for cur_cursor in children:
        if cur_cursor.kind != cindex.CursorKind.STRUCT_DECL:
            continue
        insert_start = insert_end = cur_cursor.extent.end.offset - 1
        insert_contents = "\n"
        for in_tensor in in_tensor_added:
            if in_tensor.endswith(DEQSCALE_SUFFIX):
                insert_contents += f"{indent_prefix}float {in_tensor.lower()} = 1;\n"
            else:
                insert_contents += f"{indent_prefix}int {in_tensor.lower()} = 0;\n"
        break
    updates = [(insert_start, insert_end, insert_contents)]
    return update_contents(contents, updates)


def to_quant_file_path(file_path):
    return os.path.join(os.path.dirname(file_path), "quant_" + os.path.basename(file_path))


def transform_quant(source_path):
    import stat
    from glob import glob

    check_libclang_so()
    pairs = []
    for cpp_file in glob(os.path.join(source_path, "*.cpp")):
        if "quant" in os.path.basename(cpp_file):
            continue
        hpp_file = os.path.splitext(cpp_file)[0] + ".h"
        if os.path.exists(hpp_file):
            pairs.append((cpp_file, hpp_file))

    source_files, target_files = [], []
    for cpp_file_path, hpp_file_path in pairs:
        cpp_contents, in_tensor_added = transform_quant_cpp(cpp_file_path)
        hpp_contents = transform_quant_hpp(hpp_file_path, in_tensor_added)

        target_cpp_file_path = to_quant_file_path(cpp_file_path)
        logger.info(f"\nsource cpp file: {cpp_file_path},\ntarget cpp file: {target_cpp_file_path}")

        file_permission = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
        with os.fdopen(os.open(target_cpp_file_path, os.O_CREAT | os.O_WRONLY, file_permission), 'w') as ff:
            ff.write(cpp_contents)
        source_files.append(cpp_file_path)
        target_files.append(target_cpp_file_path)

        target_hpp_file_path = to_quant_file_path(hpp_file_path)
        logger.info(f"\nsource hpp file: {hpp_file_path},\ntarget hpp file: {target_hpp_file_path}")
        with os.fdopen(os.open(target_hpp_file_path, os.O_CREAT | os.O_WRONLY, file_permission), 'w') as ff:
            ff.write(hpp_contents)
        source_files.append(hpp_file_path)
        target_files.append(target_hpp_file_path)
    
    logger.info("\nTransformed source files: [\n    " +
        "\n    ".join(source_files) + "\n]" +
        "\nTransformed target files: [\n    " +
        "\n    ".join(target_files) + "\n]"
    )
