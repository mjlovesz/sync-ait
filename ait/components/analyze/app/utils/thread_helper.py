from threading import Thread


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


def alloc_configs_for_subprocess(parallel, configs_num):
    num_process = [int(configs_num // parallel) + 1 for _ in range(configs_num % parallel)]
    num_process = num_process + [int(configs_num // parallel) for _ in range(parallel - configs_num % parallel)]
    idx = 0
    process_idx = [0]
    for num in num_process:
        process_idx.append(idx + num)
        idx += num
    return process_idx
