import os
import pandas as pd
import logging
import warnings
import secrets

from metrics.metrics import get_metric
from common.validate import validate_parameters_by_type, validate_parameters_by_func
from common.log import logger


class PermissionWarning(UserWarning):
    pass


def check_dir(output_dir):
    if not output_dir:
        raise OSError("Output directory should be empty,")

    if len(output_dir) > 4095 or any(len(item) > 255 for item in output_dir.split(r"/")):
        raise OSError("Output directory should not be too long.")
    
    if os.path.islink(output_dir):
            warnings.warn("Your attempt to use soft links as directories has triggered a security alert "
                        "and is being monitored closely for potential security threats.", PermissionWarning)

    # will not modify the original output_dir itself
    output_dir = os.path.abspath(output_dir)
    if os.path.isdir(output_dir):
        dir_stat = os.stat(output_dir)
        if dir_stat.st_uid != os.getuid():
            logger.warning("You are attempting to modify a directory that belongs to another entity.")

        if not os.access(output_dir, os.X_OK):
            raise PermissionError(f"{output_dir} is not executable for current user: Permission denied.")
        
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"{output_dir} is not writable for current user: Permission denied.")

    else:
        logger.warning("Specified directory does not exist. Trying to create...")
        os.makedirs(output_dir, 0o750, exist_ok=True)


class CaseFilter(object):

    def __init__(self):
        self._metrics = []
        self._columns = ["model input", "model output", "gold standard"]

    def add_metrics(self, **metrics):
        for metric_name, thr in metrics.items():
            self._metrics.append(get_metric(metric_name, thr))
            self._columns.append(metric_name)

    @validate_parameters_by_type(
        {
            "ins": [list],
            "outs": [list],
            "refs": [list],
        },
        in_class=True
    )
    @validate_parameters_by_func(
        {
            "ins": [len, lambda ins: all(isinstance(item, str) for item in ins)], # should not empty, should be str
            "outs": [len, lambda ins: all(isinstance(item, str) for item in ins)],
            "refs": [len, lambda ins: all(isinstance(item, str) for item in ins)],
        },
        in_class=True
    )
    def apply(self, ins, outs, refs, output_dir=None):
        if not len(ins) == len(outs) == len(refs):
            raise ValueError("Input sequences must have the same length.")

        if not self._metrics:
            raise RuntimeError("Metrics must be added first before calling apply.")

        data = dict()
        num_metrics = len(self._metrics)

        for col_idx, metric_method in enumerate(self._metrics):
            for row_idx, score in metric_method(outs, refs):
                if row_idx not in data:
                    data[row_idx] = [ins[row_idx], outs[row_idx], refs[row_idx]] + [None] * num_metrics
                    
                data[row_idx][col_idx + 3] = score

        self._save(data, output_dir)

    @validate_parameters_by_type(
        {
            "data": [dict],
            "output_dir": [str],
        }, 
        in_class=True
    )
    @validate_parameters_by_func(
        {
            "data": [],
            "output_dir": [check_dir],
        },
        in_class=True
    )
    def _save(self, data, output_dir):
        if not data:
            raise RuntimeError("The data is unexpectedly empty,")

        output_dir = os.path.abspath(output_dir)
        df = pd.DataFrame.from_dict(data, orient="index", columns=self._columns)
        output_path = os.path.join(output_dir, f"{os.getpid()}_{secrets.randbelow(100000)}.csv")
        df.to_csv(output_path, index=False)
        os.chmod(output_path, 0o640)


if __name__ == "__main__":
    case_filter = CaseFilter()
    
    case_filter.add_metrics(edit_distance=None, relative_abnormal_string_rate=None)

    ins = ["Some random tests", "我爱你中国"]
    outs = ["Some outputs", "我也爱中国"]
    refs = ["References", "我不爱国，不好意思"]

    case_filter.apply(ins, outs, refs, os.path.dirname(__file__))
