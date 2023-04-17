import os
import csv
import math
from collections import namedtuple

INVALID_ROW_VALUES = {
    "OpType": ["TransData"],
    "GroundTruth": ["*"],
    "DataType": ["NaN"],
}

MONITOR_THRESHOLD = {
    "CosineSimilarity": 0.99,
    "RelativeEuclideanDistance": 0.05,
    "KullbackLeiblerDivergence": 0.001,
    "RootMeanSquareError": 1.0,
    "MeanRelativeError": 1.0,
}

REVERSE_MONITORS = ["CosineSimilarity"]
PRINT_COLUMNS = ["Index", "OpType", "NPUDump", "GroundTruth"]

_STRATEGY_NAMES = ["FIRST_INVALID_OVERALL", "FIRST_INVALID_EACH"]
STRATEGIES = namedtuple("STRATEGIES", _STRATEGY_NAMES)(*_STRATEGY_NAMES)


def type_to_str(value_type):
    return " or ".join([ii.__name__ for ii in value_type]) if isinstance(value_type, tuple) else value_type.__name__


def check_type(value, value_type, param_name="value", additional_check_func=None, additional_msg=None):
    if not isinstance(value, value_type):
        raise TypeError(
            "{} needs to be {}, but got {}.".format(param_name, type_to_str(value_type), type(value).__name__)
        )


def check_element_type(value, element_type, param_name="value"):
    if len(value) == 0:
        raise ValueError("{} is empty".format(param_name))
    if not all([isinstance(ii, element_type) for ii in value]):
        raise ValueError("Elements in {} need to be all {}.".format(param_name, type_to_str(element_type)))


class Analyser:
    def __init__(self, csv_result_file):
        check_type(csv_result_file, str, param_name="csv_result_file")
        if not csv_result_file.endswith(".csv"):
            raise ValueError("csv_result_file not endswith csv")

        self.csv_result_file = csv_result_file
        self.monitor_threshold = {}
        for monitor, threshold in MONITOR_THRESHOLD.items():
            self.monitor_threshold[monitor] = (1 - threshold) if monitor in REVERSE_MONITORS else threshold

        self._strategy_func_dict = {
            STRATEGIES.FIRST_INVALID_OVERALL: self._first_invalid_overall,
            STRATEGIES.FIRST_INVALID_EACH: self._first_invalid_each,
        }

    def __call__(self, strategy=STRATEGIES.FIRST_INVALID_OVERALL, max_column_len=30):
        if not strategy in STRATEGIES:
            raise ValueError(f"strategy Should be one of {list(STRATEGIES)}")

        with open(self.csv_result_file, "r") as csv_file:
            self.csv_rows = [row for row in csv.DictReader(csv_file) if self._is_valid_row(row)]
        self._strategy_func = self._strategy_func_dict[strategy]
        invalid_rows, invalid_monitors = self._strategy_func()

        self._show_result(invalid_rows, invalid_monitors, max_column_len=max_column_len)
        return invalid_rows, invalid_monitors

    def _first_invalid_overall(self):
        for row in self.csv_rows:
            cur_invalid_monitors = self._get_monitors_exceeding_threshold(row, self.monitor_threshold)
            if len(cur_invalid_monitors) > 0:
                return [row], [cur_invalid_monitors]
        return [], []

    def _first_invalid_each(self):
        monitor_threshold = self.monitor_threshold.copy()  # use a copy, as will pop item later
        invalid_rows, invalid_monitors = [], []
        for row in self.csv_rows:
            cur_invalid_monitors = self._get_monitors_exceeding_threshold(row, monitor_threshold)
            if len(cur_invalid_monitors) > 0:
                invalid_rows.append(row)
                invalid_monitors.append(cur_invalid_monitors)
                for monitor in cur_invalid_monitors:
                    monitor_threshold.pop(monitor)

        return invalid_rows, invalid_monitors

    @staticmethod
    def _show_result(invalid_rows, invalid_monitors, max_column_len=30):
        if len(invalid_rows) == 0:
            return

        print("Operators may lead to inaccuracy:")
        results = {}
        for row, monitors in zip(invalid_rows, invalid_monitors):
            for monitor in monitors:
                results.setdefault("Monitor", []).append(monitor)
                results.setdefault("Value", []).append("{:.4g}".format(float(row[monitor])))

            for column in PRINT_COLUMNS:
                results.setdefault(column, []).extend([row[column]] * len(monitors))
        print_in_markdown_table(results, max_column_len=max_column_len)

    @staticmethod
    def _get_monitors_exceeding_threshold(row, monitor_threshold):
        invalid_monitors = []
        for monitor, threshold in monitor_threshold.items():
            row_value = float(row.get(monitor, "NaN"))
            if monitor in REVERSE_MONITORS:
                row_value = 1 - row_value

            if math.isnan(row_value) or math.isinf(row_value):
                continue
            if row_value > threshold:
                invalid_monitors.append(monitor)
        return invalid_monitors

    @staticmethod
    def _is_valid_row(row):
        for item_key, invalid_values in INVALID_ROW_VALUES.items():
            if row.get(item_key) in invalid_values:
                return False
        return True


def print_in_markdown_table(input_dict, max_column_len=30):
    """
    Print a dict in markdown table format.
    Dict is in format liek `{"column_1": [value1, value2], "column_2": ["value_3", "value_4"]}`.

    Exaples:
    >>> aa = {"aa": ["11", "22", "334455"], "bb": ["cc", "dd", "eeff"]}
    >>> print_in_markdown_table(aa)

    >>> # Similar with pandas function `to_markdown`
    >>> import pandas as pd
    >>> tt = pd.DataFrame(aa)
    >>> print(tt.to_markdown(index=False))
    """
    check_type(max_column_len, int, param_name="max_column_len")
    if max_column_len <= 0:
        raise ValueError(f"max_column_len needs to be > 0, but got {max_column_len}")

    check_type(input_dict, dict, param_name="input_dict")
    if len(input_dict) == 0:
        raise ValueError("input_dict is empty")
    check_element_type(input_dict.keys(), element_type=str, param_name="keys of input_dict")
    check_element_type(input_dict.values(), element_type=(list, tuple), param_name="values of input_dict")
    for key, value in input_dict.items():
        check_element_type(value, element_type=str, param_name=f"values of input_dict['{key}']")

    max_lens = {key: max([len(ii) for ii in value]) for key, value in input_dict.items()}
    max_lens = {key: min(max(value + 1, len(key) + 1), max_column_len) for key, value in max_lens.items()}

    print("|", end="")
    sep_line = "|"
    for key, max_len in max_lens.items():
        print(" " * (max_len - len(key)) + key + " |", end="")
        sep_line += "-" * max_len + ":|"
    print()
    print(sep_line)

    first_key = list(input_dict.keys())[0]
    for id in range(len(input_dict[first_key])):
        print("|", end="")
        for key, max_len in max_lens.items():
            cur_result = input_dict[key][id]
            if len(cur_result) >= max_len:
                cur_result = " " + cur_result[: max_len - 4] + "..."
            print(" " * (max_len - len(cur_result)) + cur_result + " |", end="")
        print()
    print()
