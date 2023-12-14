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

import math
import os
import random
import time
import numpy as np

from ais_bench.infer.summary import summary
from ais_bench.infer.utils import (
    get_file_content,
    get_file_datasize,
    get_fileslist_from_dir,
    list_split,
    logger,
    save_data_to_files,
)

PURE_INFER_FAKE_FILE = "pure_infer_data"
PURE_INFER_FAKE_FILE_ZERO = "pure_infer_data_zero"
PURE_INFER_FAKE_FILE_RANDOM = "pure_infer_data_random"
PADDING_INFER_FAKE_FILE = "padding_infer_fake_file"


def convert_real_files(files):
    real_files = []
    for file in files:
        if file == PURE_INFER_FAKE_FILE:
            raise RuntimeError("not support pure infer")
        elif file.endswith(".npy") or file.endswith(".NPY"):
            raise RuntimeError("not support npy file:{}".format(file))
        elif file == PADDING_INFER_FAKE_FILE:
            real_files.append(files[0])
        else:
            real_files.append(file)
    return real_files


def get_pure_infer_data(size, pure_data_type):
    lst = []
    if pure_data_type == "random":
        # random value from [0, 255]
        lst = [random.randrange(0, 256) for _ in range(size)]
    else:
        # zero value, default
        lst = [0 for _ in range(size)]

    barray = bytearray(lst)
    ndata = np.frombuffer(barray, dtype=np.uint8)
    return ndata


# get numpy array from files list combile all files
def get_narray_from_files_list(files_list, size, pure_data_type, no_combine_tensor_mode=False):
    ndatalist = []
    file_path_switch = {
        PURE_INFER_FAKE_FILE: pure_data_type,
        PURE_INFER_FAKE_FILE_ZERO: "zero",
        PURE_INFER_FAKE_FILE_RANDOM: "random",
    }
    for i, file_path in enumerate(files_list):
        logger.debug("get tensor from filepath:{} i:{} of all:{}".format(file_path, i, len(files_list)))
        if file_path_switch.get(file_path) is not None:
            ndata = get_pure_infer_data(size, file_path_switch.get(file_path))
        elif file_path == PADDING_INFER_FAKE_FILE:
            logger.debug("padding file use fileslist[0]:{}".format(files_list[0]))
            ndata = get_file_content(files_list[0])
        elif file_path is None or not os.path.exists(file_path):
            logger.error('filepath:{} not valid'.format(file_path))
            raise RuntimeError()
        else:
            ndata = get_file_content(file_path)
        ndatalist.append(ndata)
    if len(ndatalist) == 1:
        return ndatalist[0]
    else:
        ndata = np.concatenate(ndatalist)
        if not no_combine_tensor_mode and ndata.nbytes != size:
            logger.error('ndata size:{} not match {}'.format(ndata.nbytes, size))
            raise RuntimeError()
        return ndata


# get tensors from files list combile all files
def get_tensor_from_files_list(files_list, session, size, pure_data_type, no_combine_tensor_mode=False):
    ndata = get_narray_from_files_list(files_list, size, pure_data_type, no_combine_tensor_mode)
    tensor = session.create_tensor_from_arrays_to_device(ndata)
    return tensor


# Obtain filesperbatch runcount information according to file information and input description information
# The strategy is as follows:  Judge according to the realsize and file size of input 0. If the judgment fails,
# you need to force the desired value to be set
def get_files_count_per_batch(intensors_desc, fileslist, no_combine_tensor_mode=False):
    # get filesperbatch
    filesize = get_file_datasize(fileslist[0][0])
    tensorsize = intensors_desc[0].realsize
    if no_combine_tensor_mode:
        files_count_per_batch = 1
    else:
        if filesize == 0 or tensorsize % filesize != 0:
            logger.error('arg0 tensorsize: {} filesize: {} not match'.format(tensorsize, filesize))
            raise RuntimeError()
        else:
            files_count_per_batch = (int)(tensorsize / filesize)
    if files_count_per_batch == 0:
        logger.error('files count per batch is zero')
        raise RuntimeError()
    runcount = math.ceil(len(fileslist[0]) / files_count_per_batch)

    logger.info(
        "get filesperbatch files0 size:{} tensor0size:{} filesperbatch:{} runcount:{}".format(
            filesize, tensorsize, files_count_per_batch, runcount
        )
    )
    return files_count_per_batch, runcount


# Obtain tensor information and files information according to the input filelist. Create intensor form files list
# len(files_list) should equal len(intensors_desc)
def create_infileslist_from_fileslist(fileslist, intensors_desc, no_combine_tensor_mode=False):
    if len(intensors_desc) != len(fileslist):
        logger.error('fileslist:{} intensor:{} not match'.format(len(fileslist), len(intensors_desc)))
        raise RuntimeError()
    files_count_per_batch, runcount = get_files_count_per_batch(intensors_desc, fileslist, no_combine_tensor_mode)

    files_perbatch_list = [
        list(list_split(fileslist[j], files_count_per_batch, PADDING_INFER_FAKE_FILE))
        for j in range(len(intensors_desc))
    ]

    infileslist = []
    for i in range(runcount):
        infiles = []
        for j in range(len(intensors_desc)):
            logger.debug(
                "create infileslist i:{} j:{} runcount:{} lists:{} filesPerPatch:{}".format(
                    i, j, runcount, files_perbatch_list[j][i], files_count_per_batch
                )
            )
            infiles.append(files_perbatch_list[j][i])
        infileslist.append(infiles)
    return infileslist


#  outapi. Obtain tensor information and files information according to the input filelist.
#  Create intensor form files list
def create_intensors_from_infileslist(
    infileslist, intensors_desc, session, pure_data_type, no_combine_tensor_mode=False
):
    intensorslist = []
    for infiles in infileslist:
        intensors = []
        for files, intensor_desc in zip(infiles, intensors_desc):
            tensor = get_tensor_from_files_list(
                files, session, intensor_desc.realsize, pure_data_type, no_combine_tensor_mode
            )
            intensors.append(tensor)
        intensorslist.append(intensors)
    return intensorslist


def check_input_parameter(inputs_list, intensors_desc):
    if len(inputs_list) == 0:
        logger.error("Invalid args. Input args are empty")
        raise RuntimeError()
    if os.path.isfile(inputs_list[0]):
        for index, file_path in enumerate(inputs_list):
            realpath = os.readlink(file_path) if os.path.islink(file_path) else file_path
            if not os.path.isfile(realpath):
                logger.error(
                    "Invalid input args.--input:{} input[{}]:{} {} not exist".format(
                        inputs_list, index, file_path, realpath
                    )
                )
                raise RuntimeError()
    elif os.path.isdir(inputs_list[0]):
        if len(inputs_list) != len(intensors_desc):
            logger.error(
                "Invalid args. args input dir num:{0} not equal to model inputs num:{1}".format(
                    len(inputs_list), len(intensors_desc)
                )
            )
            raise RuntimeError()

        for dir_path in inputs_list:
            real_dir_path = os.readlink(dir_path) if os.path.islink(dir_path) else dir_path
            if not os.path.isdir(real_dir_path):
                logger.error("Invalid args. {} of input args is not a real dir path".format(real_dir_path))
                raise RuntimeError()
    else:
        logger.error("Invalid args. {}  of --input is invalid".format(inputs_list[0]))
        raise RuntimeError()


# outapi. get by input parameters of  inputs_List.
def create_infileslist_from_inputs_list(inputs_list, intensors_desc, no_combine_tensor_mode=False):
    check_input_parameter(inputs_list, intensors_desc)
    fileslist = []
    inputlistcount = len(inputs_list)
    intensorcount = len(intensors_desc)
    if os.path.isfile(inputs_list[0]):
        chunks = inputlistcount // intensorcount
        fileslist = list(list_split(inputs_list, chunks, PADDING_INFER_FAKE_FILE))
        logger.debug(
            "create intensors list file type inlistcount:{} intensorcont:{} chunks:{} files_size:{}".format(
                inputlistcount, intensorcount, chunks, len(fileslist)
            )
        )
    elif os.path.isdir(inputs_list[0]) and inputlistcount == intensorcount:
        fileslist = [get_fileslist_from_dir(dir) for dir in inputs_list]
        logger.debug(
            "create intensors list dictionary type inlistcount:{} intensorcont:{} files_size:{}".format(
                inputlistcount, intensorcount, len(fileslist)
            )
        )
    else:
        logger.error(
            'create intensors list filelists:{} intensorcont:{} error create'.format(inputlistcount, intensorcount)
        )
        raise RuntimeError()

    infileslist = create_infileslist_from_fileslist(fileslist, intensors_desc, no_combine_tensor_mode)
    if len(infileslist) == 0:
        logger.error('create_infileslist_from_fileslist return infileslist size: {}'.format(len(infileslist)))
        raise RuntimeError()

    return infileslist


def check_pipeline_fileslist_match_intensors(fileslist, intensors_desc):
    # check intensor amount matched
    if len(intensors_desc) != len(fileslist):
        logger.error('fileslist:{} intensor:{} not match'.format(len(fileslist), len(intensors_desc)))
        raise RuntimeError()
    # check intensor size matched
    for i, files in enumerate(fileslist):
        filesize = get_file_datasize(files[0])
        tensorsize = intensors_desc[i].realsize
        auto_mode = False
        # auto_dim_mode & auto_shape_mode are exceptional cases
        if intensors_desc[i].realsize == intensors_desc[i].size:
            if any(dim <= 0 for dim in intensors_desc[i].shape):
                auto_mode = True
        if filesize != tensorsize and not auto_mode:
            logger.error(f'tensor_num:{i} tensorsize:{tensorsize} filesize:{filesize} not match')
            raise RuntimeError()


# 不组batch的情况
def create_pipeline_fileslist_from_inputs_list(inputs_list, intensors_desc):
    check_input_parameter(inputs_list, intensors_desc)
    fileslist = []
    inputlistcount = len(inputs_list)
    intensorcount = len(intensors_desc)
    if os.path.isfile(inputs_list[0]):
        chunks = inputlistcount // intensorcount
        fileslist = list(list_split(inputs_list, chunks, PADDING_INFER_FAKE_FILE))
        logger.debug(
            f"create intensors list file type inlistcount:{inputlistcount} \
                     intensorcont:{intensorcount} chunks:{chunks} files_size:{len(fileslist)}"
        )
    elif os.path.isdir(inputs_list[0]) and inputlistcount == intensorcount:
        fileslist = [get_fileslist_from_dir(dir_) for dir_ in inputs_list]
        logger.debug(
            f"create intensors list dictionary type inlistcount:{inputlistcount} \
                     intensorcont:{intensorcount} files_size:{len(fileslist)}"
        )
    else:
        logger.error('create intensors list filelists:{inputlistcount} intensorcont:{intensorcount} error create')
        raise RuntimeError()
    try:
        check_pipeline_fileslist_match_intensors(fileslist, intensors_desc)
    except Exception as err:
        logger.error("fileslist and intensors not matched")
        raise RuntimeError from err
    infileslist = list(zip(*fileslist))
    return infileslist


def save_tensors_to_file(outputs, output_prefix, infiles_paths, outfmt, index, output_batchsize_axis):
    files_count_perbatch = len(infiles_paths[0])
    infiles_perbatch = np.transpose(infiles_paths)
    for i, out in enumerate(outputs):
        ndata = np.array(out)
        if output_batchsize_axis >= len(ndata.shape):
            logger.error(
                "error i:{0} ndata.shape:{1} len:{2} <= output_batchsize_axis:{3}  is invalid".format(
                    i, ndata.shape, len(ndata.shape), output_batchsize_axis
                )
            )
            raise RuntimeError()
        if files_count_perbatch == 1 or ndata.shape[output_batchsize_axis] % files_count_perbatch == 0:
            subdata = np.array_split(ndata, files_count_perbatch, output_batchsize_axis)
            for j in range(files_count_perbatch):
                sample_id = index * files_count_perbatch + j
                if infiles_perbatch[j][0] == PADDING_INFER_FAKE_FILE:
                    logger.debug(
                        "sampleid:{} i:{} infiles:{} is padding fake file so continue".format(
                            sample_id, i, infiles_perbatch[j]
                        )
                    )
                    continue
                file_path = os.path.join(
                    output_prefix,
                    "{}_{}.{}".format(os.path.basename(infiles_perbatch[j][0]).split('.')[0], i, outfmt.lower()),
                )
                summary.add_sample_id_infiles(sample_id, infiles_perbatch[j])
                logger.debug(
                    "save func: sampleid:{} i:{} infiles:{} outfile:{} fmt:{} axis:{}".format(
                        sample_id, i, infiles_perbatch[j], file_path, outfmt, output_batchsize_axis
                    )
                )
                summary.append_sample_id_outfile(sample_id, file_path)
                save_data_to_files(file_path, subdata[j])
        else:
            logger.error(
                'save out files error array shape:{} filesinfo:{} files_count_perbatch:{} ndata.shape\
                         {}:{}'.format(
                    ndata.shape,
                    infiles_paths,
                    files_count_perbatch,
                    output_batchsize_axis,
                    ndata.shape[output_batchsize_axis],
                )
            )
            raise RuntimeError()
