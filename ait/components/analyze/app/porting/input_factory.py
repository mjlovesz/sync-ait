from common.kit_config import InputType
from porting.cmdline_input import CommandLineInput


class InputFactory:
    """
    输入工厂类，根据输入的类型返回具体的输入子类对象
    """

    @staticmethod
    def get_input(input_type, args):
        """
        根据输入类型实例化对应的子类对象返回,目前仅规划了命令行的输入，如果后续需要扩展，添加类型和else分支.
        :param input_type: 输入类型
        :param args: 输入的初始化信息
        :return: 子类对象
        """
        if input_type == InputType.CMD_LINE:
            return CommandLineInput(args)
        elif input_type == InputType.RESTFUL:
            raise Exception('Not support yet!')
        else:
            return CommandLineInput(args)
