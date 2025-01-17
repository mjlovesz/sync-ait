import os
import string
from ait_llm.common.log import logger
from ait_llm.common.log import set_log_level

CODE_CHAR = string.printable.replace("\r", "")  # For getting rid of Chinese char and windows `\r`


def print_spelling(param, info="", level="debug"):
    param = param.get_children() if hasattr(param, "get_children") else param
    message = info + "[" + ", ".join([ii.spelling for ii in param]) + "]"
    if level.lower() == "debug":
        logger.debug(message)
    else:
        logger.info(message)


def print_update_info(insert_contents, insert_start, insert_end, cur_id=None):
    message = f"insert_start: {insert_start}, insert_end: {insert_end}, insert_contents: {insert_contents}"
    if cur_id is not None:
        message += f", cur_id: {cur_id}"
    logger.debug("Current update: " + message)


def check_libclang_so():
    import clang
    from clang import cindex

    libclang_so_path = os.path.join(os.path.dirname(clang.__file__), "native", "libclang.so")
    if os.path.exists(libclang_so_path):
        cindex.Config.set_library_file(libclang_so_path)
    else:
        loger.warning(f"libclang so: {libclang_so_path} not found, may meet error lately.")


def get_args_and_options():
    import platform    
    from clang import cindex

    ATB_HOME_PATH = os.getenv("ATB_HOME_PATH", "")
    ASCEND_TOOLKIT_HOME = os.getenv("ASCEND_TOOLKIT_HOME", "")
    ATB_SPEED_COMPILE_PATH = os.path.dirname(os.path.dirname(os.getenv("ATB_SPEED_HOME_PATH", "")))

    cur_platform = platform.machine() + "-" + platform.system()  # like "aarch64-linux"
    include_pathes = [
        os.path.join(ATB_HOME_PATH, "include"),
        os.path.join(ASCEND_TOOLKIT_HOME, cur_platform, "include"),
        ATB_SPEED_COMPILE_PATH,
        os.path.join(ATB_SPEED_COMPILE_PATH, "core", "include"),
        os.path.join(ATB_SPEED_COMPILE_PATH, "3rdparty", "nlohmannJson", "include"),
    ]
    args = ["-fsyntax-only"]
    args.extend(["-I " + include_path for include_path in include_pathes])
    options = cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    return args, options


def filter_chinese_char(contents):
    return "".join(filter(lambda ii: ii in CODE_CHAR, contents))


def update_contents(contents, updates):
    updates = sorted(updates, key=lambda xx: xx[0], reverse=True)
    for insert_start, insert_end, insert_contents in updates:
        contents = contents[:insert_start] + insert_contents + contents[insert_end:]
    return contents
