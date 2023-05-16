/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Base/ModelInfer/DynamicAippConfig.h"

namespace Base {
DynamicAippConfig::DynamicAippConfig()
{
    isActivated = false;
    modelOK = false;
};

DynamicAippConfig::~DynamicAippConfig()
{
    cropParams.clear();
    paddingParams.clear();
    dtcPixelMeanParams.clear();
    dtcPixelMinParams.clear();
    pixelVarReciParams.clear();
}

bool DynamicAippConfig::IsActivated()
{
    return isActivated;
}

bool DynamicAippConfig::ModelIsLegal()
{
    return modelOK;
}

void DynamicAippConfig::ActivateConfig()
{
    isActivated = true;
}

void DynamicAippConfig::ActivateModel()
{
    modelOK = true;
}

APP_ERROR DynamicAippConfig::SetMaxBatchSize(uint64_t maxBsParams)
{
    maxBatchSize = maxBsParams;
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetInputFormat(std::string iptFmt)
{
    inputFormat = iptFmt;
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetSrcImageSize(std::vector<int> srcImageSize)
{
    srcImageSizeW = srcImageSize[0];
    srcImageSizeH = srcImageSize[1];
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetRbuvSwapSwitch(int rsSwitch)
{
    rbuvSwapSwitch = rsSwitch;
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetAxSwapSwitch(int asSwitch)
{
    axSwapSwitch = asSwitch;
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetCscParams(std::vector<int> cscInputParams)
{
    cscParams.cscSwitch = cscInputParams[0];
    cscParams.cscMatrixR0C0 = cscInputParams[1];
    cscParams.cscMatrixR0C1 = cscInputParams[2];
    cscParams.cscMatrixR0C2 = cscInputParams[3];
    cscParams.cscMatrixR1C0 = cscInputParams[4];
    cscParams.cscMatrixR1C1 = cscInputParams[5];
    cscParams.cscMatrixR1C2 = cscInputParams[6];
    cscParams.cscMatrixR2C0 = cscInputParams[7];
    cscParams.cscMatrixR2C1 = cscInputParams[8];
    cscParams.cscMatrixR2C2 = cscInputParams[9];
    cscParams.cscOutputBias0 = cscInputParams[10];
    cscParams.cscOutputBias1 = cscInputParams[11];
    cscParams.cscOutputBias2 = cscInputParams[12];
    cscParams.cscInputBias0 = cscInputParams[13];
    cscParams.cscInputBias1 = cscInputParams[14];
    cscParams.cscInputBias2 = cscInputParams[15];
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetCropParams(std::vector<int> cropInputParams)
{
    CropParams tmpCrop;
    tmpCrop.cropSwitch = cropInputParams[0];
    tmpCrop.loadStartPosW = cropInputParams[1];
    tmpCrop.loadStartPosH = cropInputParams[2];
    tmpCrop.cropSizeW = cropInputParams[3];
    tmpCrop.cropSizeH = cropInputParams[4];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        cropParams.insert(std::make_pair(batchIndex, tmpCrop));
    }
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetPaddingParams(std::vector<int> padInputParams)
{
    PaddingParams tmpPad;
    tmpPad.paddingSwitch = padInputParams[0];
    tmpPad.paddingSizeTop = padInputParams[1];
    tmpPad.paddingSizeBottom = padInputParams[2];
    tmpPad.paddingSizeLeft = padInputParams[3];
    tmpPad.paddingSizeRight = padInputParams[4];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        paddingParams.insert(std::make_pair(batchIndex, tmpPad));
    }
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetDtcPixelMean(std::vector<int> meanInputParams)
{
    DtcPixelMean tmpMean;
    tmpMean.dtcPixelMeanChn0 = meanInputParams[0];
    tmpMean.dtcPixelMeanChn1 = meanInputParams[1];
    tmpMean.dtcPixelMeanChn2 = meanInputParams[2];
    tmpMean.dtcPixelMeanChn3 = meanInputParams[3];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        dtcPixelMeanParams.insert(std::make_pair(batchIndex, tmpMean));
    }
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetDtcPixelMin(std::vector<float> minInputParams)
{
    DtcPixelMin tmpMin;
    tmpMin.dtcPixelMinChn0 = minInputParams[0];
    tmpMin.dtcPixelMinChn1 = minInputParams[1];
    tmpMin.dtcPixelMinChn2 = minInputParams[2];
    tmpMin.dtcPixelMinChn3 = minInputParams[3];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        dtcPixelMinParams.insert(std::make_pair(batchIndex, tmpMin));
    }
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetPixelVarReci(std::vector<float> reciInputParams)
{
    PixelVarReci tmpReci;
    tmpReci.dtcPixelVarReciChn0 = reciInputParams[0];
    tmpReci.dtcPixelVarReciChn1 = reciInputParams[1];
    tmpReci.dtcPixelVarReciChn2 = reciInputParams[2];
    tmpReci.dtcPixelVarReciChn3 = reciInputParams[3];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        pixelVarReciParams.insert(std::make_pair(batchIndex, tmpReci));
    }
    return APP_ERR_OK;
}

uint64_t DynamicAippConfig::GetMaxBatchSize()
{
    return maxBatchSize;
}

std::string DynamicAippConfig::GetInputFormat()
{
    return inputFormat;
}

int32_t DynamicAippConfig::GetSrcImageSizeW()
{
    return srcImageSizeW;
}

int32_t DynamicAippConfig::GetSrcImageSizeH()
{
    return srcImageSizeH;
}

int8_t DynamicAippConfig::GetRbuvSwapSwitch()
{
    return rbuvSwapSwitch;
}

int8_t DynamicAippConfig::GetAxSwapSwitch()
{
    return axSwapSwitch;
}

CscParams DynamicAippConfig::GetCscParams()
{
    return cscParams;
}

std::unordered_map<uint64_t, CropParams> DynamicAippConfig::GetCropParams()
{
    return cropParams;
}

std::unordered_map<uint64_t, PaddingParams> DynamicAippConfig::GetPaddingParams()
{
    return paddingParams;
}

std::unordered_map<uint64_t, DtcPixelMean> DynamicAippConfig::GetDtcPixelMean()
{
    return dtcPixelMeanParams;
}

std::unordered_map<uint64_t, DtcPixelMin> DynamicAippConfig::GetDtcPixelMin()
{
    return dtcPixelMinParams;
}

std::unordered_map<uint64_t, PixelVarReci> DynamicAippConfig::GetPixelVarReci()
{
    return pixelVarReciParams;
}
}   // namespace Base