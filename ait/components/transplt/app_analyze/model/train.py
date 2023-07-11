import os
from app_analyze.model.api_seq_project import APISeqProject
from app_analyze.porting.input_factory import InputFactory
from app_analyze.common.kit_config import InputType, Args
from app_analyze.scan.sequence.seq_handler import SeqHandler
from app_analyze.utils import log_util


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

    def train(self, path):
        dataset = self._load_data(path)
        api_seqs = self._scan_sources(dataset)
        apis = SeqHandler.cluster_api_seqs(api_seqs)
        print()


if __name__ == '__main__':
    model = Model()
    # "/home/liuzhe/package/opencv-4.5.4/samples/cpp"
    model.train("/home/liuzhe/samples/opencv")
