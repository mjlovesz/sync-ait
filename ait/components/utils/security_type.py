# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import re


STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")


def type_to_str(value_type):
    return ' or '.join([ii.__name__ for ii in value_type]) if isinstance(value_type, tuple) else value_type.__name__


def check_type(value, value_type, param_name="value", additional_check_func=None, additional_msg=None):
    if not isinstance(value, value_type):
        raise TypeError('{} must be {}, not {}.'.format(param_name, type_to_str(value_type), type(value).__name__))
    if additional_check_func is not None:
        additional_msg = (" " + additional_msg) if additional_msg else ""
        if isinstance(value, (list, tuple)):
            if not all(list(map(additional_check_func, value))):
                raise ValueError("Element in {} is invalid.".format(param_name) + additional_msg)
        elif not additional_check_func(value):
            raise ValueError("Value of {} is invalid.".format(param_name) + additional_msg)


def check_number(value, value_type=(int, float), min_value=None, max_value=None, param_name="value"):
    check_type(value, value_type, param_name=param_name)
    if max_value is not None and value > max_value:
        raise ValueError("{} = {} is larger than {}.".format(param_name, value, max_value))
    if min_value is not None and value < min_value:
        raise ValueError("{} = {} is smaller than {}.".format(param_name, value, min_value))


def check_int(value, min_value=None, max_value=None, param_name="value"):
    check_number(value, value_type=int, min_value=min_value, max_value=max_value, param_name=param_name)


def check_element_type(value, element_type, value_type=(list, tuple), param_name="value"):
    check_type(
        value=value,
        value_type=value_type,
        param_name=param_name,
        additional_check_func=lambda xx: isinstance(xx, element_type),
        additional_msg="Should be all {}.".format(type_to_str(element_type)),
    )


def check_character(value, param_name="value"):
    max_depth = 100

    def check_character_recursion(inner_value, depth=0):
        if isinstance(inner_value, str):
            if re.search(STR_WHITE_LIST_REGEX, inner_value):
                raise ValueError("{} contains invalid characters.".format(param_name))
        elif isinstance(inner_value, (list, tuple)):
            if depth > max_depth:
                raise ValueError("Recursion depth of {} exceeds limitation.".format(param_name))

            for sub_value in inner_value:
                check_character_recursion(sub_value, depth=depth + 1)

    check_character_recursion(value)


def check_dict_character(dict_value, key_max_len=512, param_name="dict"):
    max_depth = 100

    def check_dict_character_recursion(inner_dict_value, depth=0):
        check_type(inner_dict_value, dict, param_name=param_name)

        for key, value in inner_dict_value.items():
            key = str(key)
            check_character(key, param_name=f"{param_name} key")
            if key_max_len > 0 and len(key) > key_max_len:
                raise ValueError("Length of {} key exceeds limitation {}.".format(param_name, key_max_len))
            if isinstance(value, dict):
                if depth > max_depth:
                    raise ValueError("Recursion depth of {} exceeds limitation.".format(param_name))
                check_dict_character_recursion(value, depth=depth + 1)
            else:
                check_character(value, param_name=param_name)

    check_dict_character_recursion(dict_value)