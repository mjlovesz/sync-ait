# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
import json
import argparse
import numpy as np


def get_topk_list(k, origin_list):
    temp = sorted(origin_list)[-k:]
    temp.reverse()
    res = []
    logging.info("temp:", temp)
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
    logging.info("k Maximum with indices : ", str(topk_list))
    logging.info("infer count:{len(times)} mean:{np.mean(times)} max:{np.max(times)} min:{np.min(times)}")
    if np.min(times) != 0:
        logging.info("max-min  rate:{(np.max(times) - np.min(times)) * 100.0 / np.min(times)}% ")
    if np.mean(times) != 0:
        logging.info("max-mean rate:{(np.max(times) - np.mean(times)) * 100.0 / np.mean(times)}%")
    topk_index = [ i[0] for i in topk_list ]
    logging.info(topk_index)
    if args.output is not None:
        with open("{args.output}/topk_index.json", 0640) as f:
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
        logging.info("error mode:{input_args.mode}")
        sys.exit(-1)