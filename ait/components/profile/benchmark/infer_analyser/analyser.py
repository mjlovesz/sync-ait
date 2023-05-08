
import os
import sys
import json
import argparse
import numpy as np


def get_topk_list(k, origin_list):
    temp = sorted(origin_list)[-k:]
    temp.reverse()
    res = []
    print("temp:", temp)
    for ele in temp:
        res.append((origin_list.index(ele), ele))
    return res


def get_file_info(file):
    info = None
    with open(file, 'rb') as f:
        info = json.load(f)
    return info


def analyse_plog(args):
    info = get_file_info(args.summary_path)
    

def analyse_topk_times(args):
    info = get_file_info(args.summary_path)

    times = info["npu_compute_time_list"]
    k = 5
    topk_list = get_topk_list(k, times)
    print("k Maximum with indices : " + str(topk_list))
    print("infer count:{} mean:{} max:{} min:{}".format(len(times), np.mean(times), np.max(times), np.min(times)))
    if np.min(times) > 0:
        print("max-min  rate:{}% ".format((np.max(times) - np.min(times)) * 100.0 / np.min(times)))
    if np.mean(times) > 0:
        print("max-mean rate:{}%".format((np.max(times) - np.mean(times)) * 100.0 / np.mean(times)))
    topk_index = [ i[0] for i in topk_list ]
    print(topk_index)
    if args.output is not None:
        with open("{}/topk_index.json".format(args.output), "w") as f:
            f.write(json.dumps(topk_index))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", help="the sumary path")
    parser.add_argument("--plog", help="plog path")
    parser.add_argument("--output", default=None, help="the output path")
    parser.add_argument("--mode", default="times", choices=["times", "plog"], help="mode (times or plog)")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    input_args = get_args()
    if input_args.mode == "times":
        analyse_topk_times(input_args)
    elif input_args.mode == "plog":
        analyse_plog(input_args)
    else:
        print("error mode:{}".format(input_args.mode))
        sys.exit(-1)