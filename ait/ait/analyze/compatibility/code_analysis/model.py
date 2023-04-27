# Copyright 2023 Huawei Technologies Co., Ltd
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
import utils
import re
import math
import pandas as pd


analysis_apis = {
    "Crop": "acldvppVpcCropAsync",
    "Resize": "acldvppVpcResizeAsync",
    "CropResize": "acldvppVpcCropResizeAsync",
    "CropPaste": "acldvppVpcCropAndPasteAsync",
    "CropResizePaste": "acldvppVpcCropResizePasteAsync",
    "MakeBorder": "acldvppVpcMakeBorderAsync",
    "CropBatch": "acldvppVpcBatchCropAsync",
    "CropResizeBatch": "acldvppVpcBatchCropResizeAsync",
    "CropPasteBatch": "acldvppVpcBatchCropAndPasteAsync",
    "CropResizePasteBatch": "acldvppVpcBatchCropResizePasteAsync",
    "MakeBorderBatch": "acldvppVpcBatchCropResizeMakeBorderAsync",
    "VdecSF": "aclvdecSendFrame",
    "VdecSkip": "aclvdecSendSkippedFrame",
    "VdecCCA": "acldvppVpcConvertColorAsync",
    "AippInputFormat_py": "acl.mdl.set_aipp_input_format",
    "AippInputFormat_cpp": "aclmdlSetAIPPInputFormat",
    "AippCscParams_py": "acl.mdl.set_aipp_csc_params",
    "AippCscParams_cpp": "aclmdlSetAIPPCscParams"
}


def Evaluate(path):
    prof_path = os.path.join(path, 'profiling')
    if not os.path.isdir(prof_path):
        print('[error] profiling path not exist, {prof_path}.')
        return
    prof_path = utils.check_profiling_data(prof_path)

    # dvpp vpc接口选择和优化
    analyze_dvpp_vpc(prof_path, analysis_apis)

    # dvpp vdec接口选择和优化
    analyze_dvpp_vdec(prof_path, analysis_apis)


def analyze_dvpp_vpc(profiling_path, API):
    acl_statistic_data = utils.get_statistic_profile_data(profiling_path)
    data = pd.read_csv(acl_statistic_data)
    countCrop = 0
    countResize = 0
    countCropResize = 0
    countCropPaste = 0
    countCropResizePaste = 0
    countMakeBorder = 0
    for line in data.itertuples():
        if (line[1] == API["Crop"]):
            countCrop = line[5]
        if (line[1] == API["Resize"]):
            countResize = line[5]
        if (line[1] == API["CropResize"]):
            countCropResize = line[5]
        if (line[1] == API["CropPaste"]):
            countCropPaste = line[5]
        if (line[1] == API["CropResizePaste"]):
            countCropResizePaste = line[5]
        if (line[1] == API["MakeBorder"]):
            countMakeBorder = line[5]
    if (countCrop != 0 or countResize != 0 or countCropResize !=
            0 or countCropPaste != 0 or countCropResizePaste != 0 or countMakeBorder != 0):
        if (countCrop >= 2):
            print(f'检测到使用{API["Crop"]}接口，循环处理图片，建议使用{API["CropBatch"]}接口。')
        if (countCrop >= 2 and countResize >= 2 and countCrop == countResize):
            print(f'检测到连续调用{API["Crop"]}和{API["Resize"]}接口，同时循环处理多张图，建议使用{API["CropResizeBatch"]}接口。')
        if (countCropResize >= 2):
            print(f'检测到连续调用{API["CropResize"]}接口，建议使用{API["CropResizeBatch"]}接口。')
        if (countCropPaste >= 2):
            print(f'检测到循环调用{API["CropPaste"]}接口，建议使用{API["CropPasteBatch"]}接口。')
        if (countCropResizePaste >= 2):
            print(f'检测到循环调用{API["CropResizePaste"]}接口，建议使用{API["CropResizePasteBatch"]}接口。')
        if (countMakeBorder >= 2 and (countCrop != 0 or countResize != 0) and \
            (countMakeBorder == countCrop or countMakeBorder == countResize)):
            print(f'检测到循环调用{API["Crop"]}和{API["Resize"]}和{API["MakeBorder"]}接口，建议使用{API["MakeBorderBatch"]}接口。')
    else:
        print("在这个AI处理器上，可能没有使用VPCAPI接口，因而在这个方向上，知识库暂时没有调优建议。")


def analyze_dvpp_vdec(profiling_path, API):
    acl_statistic_data = utils.get_statistic_profile_data(profiling_path)
    data = pd.read_csv(acl_statistic_data)
    countVpcCCA = 0
    countVdecSF = 0
    for line in data.itertuples():
        if (line[1] == API["VdecCCA"]):
            countVpcCCA = line[5]
        if (line[1] == API["VdecSF"]):
            countVdecSF = line[5]
    if (countVpcCCA != 0 or countVdecSF != 0):
        if (countVpcCCA >= 1 & countVdecSF == 0):
            print(f'检测使用了{API["VdecCCA"]}接口。')
            print(f'如果您使用的是昇腾710 AI处理器，该处理器视频解码接口{API["VdecSF"]}' \
                '支持输出YUV420SP格式或RGB888格式，可设置接口参数输出不同的格式，' \
                '建议省去调用{API["VdecCCA"]}进行格式转换的步骤，减少接口调用。')
            print(
                f'同时，在视频解码+模型推理的场景下，若视频的帧数比较多，且不是每一帧都需要进行推理，对于不需要推理的帧，' \
                '推荐您使用{API["VdecSkip"]}接口进行解码，不输出解码结果。')
        if (countVdecSF >= 1 & countVpcCCA == 0):
            print(f'检测使用了{API["VdecSF"]}接口。')
            print('在昇腾710 AI处理器上，VPC图像处理功能支持输出YUV400格式（灰度图像）,' \
                '如果模型推理的输入图像是灰度图像，建议您直接使用VPC功能，无需再使用AIPP色域转换功能。')
            print(
                '同时，在视频解码+模型推理的场景下，若视频的帧数比较多，且不是每一帧都需要进行推理，对于不需要推理的帧，' \
                '推荐您使用{API["VdecSkip"]}接口进行解码，不输出解码结果。')
        if (countVpcCCA >= 1 & countVdecSF >= 1):
            print(f'检测同时使用了{API["VdecCCA"]}接口以及{API["VdecSF"]}接口。')
            print(f'在昇腾710 AI处理器上，视频解码接口{API["VdecSF"]}' \
                '支持输出YUV420SP格式或RGB888格式，可设置接口参数输出不同的格式，' \
                '建议省去调用{API["VdecCCA"]}进行格式转换的步骤，减少接口调用。')
            print(
                f'同时，在视频解码+模型推理的场景下，若视频的帧数比较多，且不是每一帧都需要进行推理，对于不需要推理的帧，' \
                '推荐您使用{API["VdecSkip"]}接口进行解码，不输出解码结果。')
    else:
        print('在此 AI 处理器上，并没有使用到 VDECAPI接口。所以在这个方向上，知识库并没有调优建议。')
        print(f'但是在视频解码+模型推理的场景下，如果用户视频的帧数很大并且不是每一帧都需要推断，' \
            '建议您使用{API["VdecSkip"]}接口以提升使用体验。')

