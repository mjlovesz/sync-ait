
import os
import sys
import json


def get_times_list(file):
    time_list = []
    with open(file, 'rb') as f:
        for line in f.readlines():
            s = line[0:-3]
            value = float(s)
            time_list.append(value)
    return time_list


def get_pid(file):
    pid = None
    if not os.path.exists(file):
        print("{} file not exist".format(file))
    else:
        with open(file, 'rb') as fd:
            pid = int(fd.read())
    return pid


if __name__ == '__main__':
    times_file = sys.argv[1]
    pid_file = sys.argv[2]
    out_file = sys.argv[3]

    times = get_times_list(times_file)
    t_pid = get_pid(pid_file)
    info = {"pid": t_pid, "npu_compute_time_list": times}
    with open(os.path.join(out_file), 'w') as ff:
        json.dump(info, ff)

