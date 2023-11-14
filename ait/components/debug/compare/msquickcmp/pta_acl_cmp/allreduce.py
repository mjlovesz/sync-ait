# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
import stat
import csv
import logging
import argparse

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm

from msquickcmp.pta_acl_cmp.utils import TensorBinFile
from msquickcmp.pta_acl_cmp.cmp_algorithm import cosine_similarity, max_relative_error, mean_relative_error, relative_euclidean_distance

WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def cosine_similarity(mat0: NDArray, mat1: NDArray) -> float:
    try:
        m0 = np.ndarray.flatten(mat0) / norm(mat0)
    except ZeroDivisionError as err:
        raise RuntimeError("get ZeroDivisionError when calc cosine_similarity") from err
    try:
        m1 = np.ndarray.flatten(mat1) / norm(mat1)
    except ZeroDivisionError as err:
        raise RuntimeError("get ZeroDivisionError when calc cosine_similarity") from err
    return np.dot(m0, m1)


def compare_allreduce(root_dir_0, root_dir_1, csv_output_path):
    """
    Function:
        Compare the output of Allreduce with gold data
        genrate a xlsx file
    Arg:
        root_dir_0: root dir for process 0
        root_dir_1  root dir for process 1
    """
    dir_path_list = []
    result_process_0 = []
    result_process_1 = []

    process_0 = os.path.basename(os.path.normpath(root_dir_0))
    process_1 = os.path.basename(os.path.normpath(root_dir_1))

    # 获取所有需要比对的路径
    for dirpath, dirnames, _ in os.walk(root_dir_0):
        for dirname in dirnames:
            if dirname.endswith('1_AllReduceHcclRunner'):
                dir_path = os.path.join(dirpath, dirname)
                dir_path_list.append(dir_path)

    for dir_path in dir_path_list:
        input_tensor_0_path = os.path.join(dir_path, "before/intensor0.bin")
        input_tensor_1_path = input_tensor_0_path.replace(process_0, process_1)
        output_tensor_0_path = os.path.join(dir_path, "after/outtensor0.bin")
        output_tensor_1_path = output_tensor_0_path.replace(process_0, process_1)

        gold = TensorBinFile(input_tensor_0_path).get_data() + TensorBinFile(input_tensor_1_path).get_data()
        output_0 = TensorBinFile(output_tensor_0_path).get_data()
        output_1 = TensorBinFile(output_tensor_1_path).get_data()
        result_process_0.append([dir_path,
                                 cosine_similarity(gold, output_0), 
                                 max_relative_error(gold, output_0),
                                 mean_relative_error(gold, output_0), 
                                 relative_euclidean_distance(gold, output_0),
                                ]
                                )
        
        result_process_1.append([dir_path.replace(process_0, process_1),
                                 cosine_similarity(gold, output_1), 
                                 max_relative_error(gold, output_1),
                                 mean_relative_error(gold, output_1), 
                                 relative_euclidean_distance(gold, output_1),
                                ]
                                )
        
        allreduce_compare_result = os.path.join(csv_output_path, "allreduce_compare_result.csv")

    if os.path.exists(allreduce_compare_result):
        logging.error("Error, file already exists!")
        os.remove(allreduce_compare_result)
    
    with os.fdopen(os.open(allreduce_compare_result, WRITE_FLAGS, WRITE_MODES), 'w',
                                   newline="") as fp_write:
        writer = csv.writer(fp_write)
        writer.writerow(["allreduce", "cosine_similarity", "max_relative_error", "mean_relative_error", "relative_euclidean_distance"])
        writer.writerows(result_process_0)
        writer.writerows(result_process_1)

    logging.info("The comparison results are generated in %s", allreduce_compare_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_0_path",
                        type=str,
                        help="the path of process 0")

    parser.add_argument("--process_1_path",
                        type=str,
                        help="the path of process 1")

    parser.add_argument("--output_path",
                        type=str,
                        help="the output path of csv")
    args = parser.parse_args()

    process_0_path = args.process_0_path
    process_1_path = args.process_1_path
    output_path = args.output_path

    compare_allreduce(process_0_path, process_1_path, output_path)