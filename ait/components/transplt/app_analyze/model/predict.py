import os
import time

from app_analyze.model.api_seq_project import APISeqProject
from app_analyze.porting.input_factory import InputFactory
from app_analyze.common.kit_config import InputType, Args, KitConfig
from app_analyze.scan.sequence.seq_handler import load_api_seqs, set_idx_tbl, set_expert_libs_tbl
from app_analyze.utils import log_util
from app_analyze.utils.log_util import logger


class Model:

    @staticmethod
    def _load_data():
        # load expert libs and idx dict
        idx_path = './seqs_idx.bin'
        if not os.path.exists(idx_path):
            raise Exception('Source directory is not existed!')

        idx_seq_dict = load_api_seqs(idx_path)
        set_idx_tbl(idx_seq_dict)

        expert_libs_path = './expert_libs.bin'
        expert_libs = load_api_seqs(expert_libs_path)
        set_expert_libs_tbl(expert_libs)

    @staticmethod
    def _scan_sources(path):
        logger_level = 'INFO'
        log_util.set_logger_level(logger_level)
        log_util.init_file_logger()

        args = Args(path, 'json', logger_level, 'cmake')
        inputs = InputFactory.get_input(InputType.CUSTOM, args)
        inputs.resolve_user_input()

        project = APISeqProject(inputs, train_flag=False)
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
    '/home/liuzhe/package/opencv-4.5.4/samples/cpp'
    # '/home/liuzhe/samples/opencv'
    # model.train(samples='/home/liuzhe/samples/api-union-test')
    # model.train(seqs='./mxbase.seqs.bin', seqs_idx='./mxbase.seqs_idx.bin')
    model.predict('/home/liuzhe/package/opencv-4.5.4/samples/cpp')
