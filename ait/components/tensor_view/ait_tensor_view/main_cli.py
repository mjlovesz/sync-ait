import argparse
import os

from operation import SliceOperation, PermuteOperation
from handler import handle_tensor_view
from components.utils.parser import BaseCommand
from components.utils.file_open_check import FileStat


def check_output_path_legality(value):
    if not value:
        return value
    path_value = value
    try:
        file_stat = FileStat(path_value)
    except Exception as err:
        raise argparse.ArgumentTypeError(f"output path:{path_value} is illegal. Please check.") from err
    if not file_stat.is_basically_legal("write", strict_permission=False):
        raise argparse.ArgumentTypeError(f"output path:{path_value} can not write. Please check.")
    return path_value


def check_input_path_legality(value):
    if not value:
        return value
    inputs_list = value.split(',')
    for input_path in inputs_list:
        try:
            file_stat = FileStat(input_path)
        except Exception as err:
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.") from err
        if not file_stat.is_basically_legal('read', strict_permission=False):
            raise argparse.ArgumentTypeError(f"input path:{input_path} is illegal. Please check.")
    return value


def parse_operations(value):
    operations = value.split(";")

    ops = []

    for op in operations:
        if op.startswith("[") and op.endswith("]"):
            ops.append(SliceOperation(op))
        elif op.startswith("(") and op.endswith(")"):
            ops.append(PermuteOperation(op))
        else:
            raise SyntaxError(f"Invalid operation string: {op}")

    return ops


def get_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


class TensorViewCommand(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            "--bin", "-b",
            type=check_input_path_legality,
            required=True,
            help="Bin file path"
        )

        parser.add_argument(
            "--print", "-p",
            action="store_true",
            help="print tensor"
        )

        parser.add_argument(
            "--output", "-o",
            type=check_output_path_legality,
            default=os.getcwd(),
            nargs="?",
            help="where the tensor should be saved (default: current directory)"
        )

        parser.add_argument(
            "--operations", "-op",
            type=parse_operations,
            help="Each operation is separated by a semicolon; slice operations should use square brackets; permute "
                 "sequences should use parentheses"
        )

    def handle(self, args):
        handle_tensor_view(args)


def get_cmd_instance():
    help_info = ""
    cmd_instance = TensorViewCommand("tensor-view", help_info)
    return cmd_instance
