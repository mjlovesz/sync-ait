import os
import time

from app_analyze.model.api_seq_project import APISeqProject
from app_analyze.porting.input_factory import InputFactory
from app_analyze.common.kit_config import InputType, Args, KitConfig
from app_analyze.scan.sequence.seq_handler import load_api_seqs, mining_api_seqs, get_idx_tbl
from app_analyze.utils import log_util
from app_analyze.utils.log_util import logger


class Model:

    @staticmethod
    def _load_data(path):
        if not os.path.exists(path):
            raise Exception(f'{path} is not existed!')

        rst = []
        for item in os.scandir(path):
            if item.is_dir():
                rst.append(item.path)
            elif item.is_file():
                rst.append(item.path)
        return rst

    @staticmethod
    def _scan_sources(files):
        logger_level = 'INFO'
        log_util.set_logger_level(logger_level)
        log_util.init_file_logger()

        api_seqs = []
        for path in files:
            args = Args(path, 'json', logger_level, 'cmake')
            inputs = InputFactory.get_input(InputType.CUSTOM, args)
            inputs.resolve_user_input()

            project = APISeqProject(inputs)
            project.setup_file_matrix()
            project.setup_scanners()
            project.scan()
            api_seqs += project.get_api_seqs()

        return api_seqs

    def train(self, **kwargs):
        start = time.time()

        api_seqs = seqs = None
        idx_seq_dict = None
        samples = kwargs.get('samples', None)
        if samples:
            if not os.path.exists(samples):
                raise Exception('Source directory is not existed!')
            KitConfig.SOURCE_DIRECTORY = os.path.abspath(samples)

            dataset = self._load_data(samples)
            api_seqs = self._scan_sources(dataset)
            eval_time = time.time() - start
            logger.info(f'The time of getting sequences is {eval_time}s.')
            seqs = mining_api_seqs(api_seqs)
            idx_seq_dict = get_idx_tbl()
            eval_time = time.time() - start
            logger.info(f'Total time is {eval_time}.')
        else:
            seqs_file = kwargs.get('seqs', None)
            seqs_idx_file = kwargs.get('seqs_idx', None)
            if seqs_file and seqs_idx_file:
                api_seqs = load_api_seqs(seqs_file)
                idx_seq_dict = load_api_seqs(seqs_idx_file)
                seqs = mining_api_seqs(api_seqs, idx_seq_dict)
                eval_time = time.time() - start
                logger.info(f'Total time is {eval_time}.')

        return api_seqs, seqs, idx_seq_dict


if __name__ == '__main__':
    model = Model()
    # '/home/liuzhe/package/opencv-4.5.4/samples/cpp'
    # '/home/liuzhe/samples/opencv'
    model.train(samples='/home/liuzhe/samples/api-union-test')
    # model.train(seqs='./mxbase.seqs.bin', seqs_idx='./mxbase.seqs_idx.bin')
    # model.train(samples='/home/liuzhe/package/opencv-4.5.4/samples/cpp')
