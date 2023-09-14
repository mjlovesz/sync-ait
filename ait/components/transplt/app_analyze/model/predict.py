import os
import time
import json

from app_analyze.model.seq_project import SeqProject
from app_analyze.porting.input_factory import InputFactory
from app_analyze.common.kit_config import InputType, Args, KitConfig
from app_analyze.scan.sequence.seq_handler import load_api_seqs
from app_analyze.scan.sequence.acc_libs import set_expert_libs
from app_analyze.scan.sequence.seq_desc import set_api_lut
from app_analyze.utils import log_util
from app_analyze.utils.log_util import logger


class Model:

    @staticmethod
    def _load_data():
        # load expert libs and idx dict
        all_idx_dict = dict()
        idx_path = ['./opencv.lut.bin', './mxbase.lut.bin']  #
        for val in idx_path:
            if not os.path.exists(val):
                raise Exception(f'File {val} is not existed!')

            idx_seq_dict = load_api_seqs(val)
            all_idx_dict.update(idx_seq_dict)

        set_api_lut(all_idx_dict)

        # expert_libs_path = './expert_libs.bin'
        # expert_libs = load_api_seqs(expert_libs_path)
        with open('./expert_libs.json', 'r') as f:
            expert_libs = json.load(f)
            set_expert_libs(expert_libs)

    @staticmethod
    def _scan_sources(path):
        logger_level = 'INFO'
        log_util.set_logger_level(logger_level)
        log_util.init_file_logger()

        args = Args(path, 'csv', logger_level, 'cmake')
        inputs = InputFactory.get_input(InputType.CUSTOM, args)
        inputs.resolve_user_input()

        project = SeqProject(inputs, train_flag=False)
        project.setup_reporters(None)
        project.setup_file_matrix()
        project.setup_scanners()
        project.scan()

    def predict(self, samples):
        start = time.time()
        if not os.path.exists(samples):
            raise Exception('Source directory is not existed!')
        KitConfig.SOURCE_DIRECTORY = os.path.abspath(samples)

        self._load_data()
        self._scan_sources(KitConfig.SOURCE_DIRECTORY)
        eval_time = time.time() - start
        logger.info(f'The time of getting sequences is {eval_time}s.')
        eval_time = time.time() - start
        logger.info(f'Total time is {eval_time}.')


if __name__ == '__main__':
    model = Model()
    # '/home/liuzhe/package/opencv-4.5.4/samples/cpp'
    # /home/liuzhe/samples/gpu_mat_test
    # /home/liuzhe/samples/HyperVID/Prj-Cpp
    model.predict('/home/liuzhe/samples/gpu_mat_test')
