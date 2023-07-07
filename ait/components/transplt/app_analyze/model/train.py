import os
from app_analyze.model.api_seq_project import APISeqProject
from app_analyze.porting.input_factory import InputFactory
from app_analyze.common.kit_config import InputType, Args


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
        for path in files:
            args = Args(path, 'json', 'INFO', 'cmake')
            inputs = InputFactory.get_input(InputType.CUSTOM, args)
            inputs.resolve_user_input()

            project = APISeqProject(inputs)
            project.setup_file_matrix()
            project.setup_scanners()
            project.scan()

    def train(self, path):
        rst = self._load_data(path)
        self._scan_sources(rst)


if __name__ == '__main__':
    model = Model()
    model.train("/home/liuzhe/package/opencv-4.5.4/samples/cpp")
