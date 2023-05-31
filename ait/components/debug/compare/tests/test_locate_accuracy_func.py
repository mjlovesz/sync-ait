

import logging
import sys
import os
import pytest
from msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msquickcmp.accuracy_locat.locat_accuracy import find_accuracy_interval

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    # staticmethod or classmethod
    @classmethod
    def get_base_path(cls):
        _current_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(_current_dir, "../test/testdata/")

    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    def init(self):
        self.args_1 = CmpArgsAdapter(
            os.path.join(self.get_base_path(), 'model/output1.onnx'), # gold_model
            os.path.join(self.get_base_path(), 'model/output1.om'), # om_model
            "", # input_data_path
            "/usr/local/Ascend/ascend-toolkit/latest/", # cann_path
            os.path.join(self.get_base_path(), '/test/output/'), # out_path
            "", # input_shape
            "0", # device
            "", # output_size
            "", # output_nodes
            False, # advisor
            "", # dym_shape_range
            True, # dump
            False # bin2npy
        )
        self.model_1_name = ""

 # testcases

    def test_basic_func(self):
        logger.info(self.args_1.model_path)
        logger.info(self.args_1.offline_model_path)
        logger.info(self.args_1.input_path)
        logger.info(self.args_1.cann_path)
        logger.info(self.args_1.out_path)
        logger.info(self.args_1.input_shape)
        logger.info(self.args_1.device)
        logger.info(self.args_1.output_size)
        logger.info(self.args_1.output_nodes)
        logger.info(self.args_1.advisor)
        logger.info(self.args_1.dym_shape_range)
        logger.info(self.args_1.dump)
        logger.info(self.args_1.bin2npy)
        # find_accuracy_interval(self.args_1, "endnode_name", "")
