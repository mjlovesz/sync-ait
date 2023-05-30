

import pytest
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.accuracy_locat.locat_accuracy import find_accuracy_interval


class TestClass:
    # staticmethod or classmethod

    def init(self):
        self.arg_1 = CmpArgsAdapter(
                                    './output.onnx', # gold_model
                                    './output.om', # om_model
                                    "", # input_data_path
                                    "/usr/local/Ascend/ascend-toolkit/latest/", # cann_path
                                    '/root/miniconda3/envs/tjh/ait/ait/components/debug/compare/result/test/20230530181906/',
                                    "",
                                    "0",
                                    "",
                                    "",
                                    False,
                                    "",
                                    True,
                                    False
        )



    # testcases