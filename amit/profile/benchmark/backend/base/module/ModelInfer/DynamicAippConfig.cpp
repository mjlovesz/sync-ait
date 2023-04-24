/*
 * Copyright(C) 2020. Huawei Technologies Co.,Ltd. All rights reserved.
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

APP_ERROR DynamicAippConfig::SetCscParams(std::vector<int> cscPrm)
{
    cscParams.cscSwitch = cscPrm[0];
    cscParams.cscMatrixR0C0 = cscPrm[1];
    cscParams.cscMatrixR0C1 = cscPrm[2];
    cscParams.cscMatrixR0C2 = cscPrm[3];
    cscParams.cscMatrixR1C0 = cscPrm[4];
    cscParams.cscMatrixR1C1 = cscPrm[5];
    cscParams.cscMatrixR1C2 = cscPrm[6];
    cscParams.cscMatrixR2C0 = cscPrm[7];
    cscParams.cscMatrixR2C1 = cscPrm[8];
    cscParams.cscMatrixR2C2 = cscPrm[9];
    cscParams.cscOutputBias0 = cscPrm[10];
    cscParams.cscOutputBias1 = cscPrm[11];
    cscParams.cscOutputBias2 = cscPrm[12];
    cscParams.cscInputBias0 = cscPrm[13];
    cscParams.cscInputBias1 = cscPrm[14];
    cscParams.cscInputBias2 = cscPrm[15];
    return APP_ERR_OK;
}
    
APP_ERROR DynamicAippConfig::SetCropParams(std::vector<int> cropPrm)
{
    CropParams tmpCrop;
    tmpCrop.cropSwitch = cropPrm[0];
    tmpCrop.loadStartPosW = cropPrm[1];
    tmpCrop.loadStartPosH = cropPrm[2];
    tmpCrop.cropSizeW = cropPrm[3];
    tmpCrop.cropSizeH = cropPrm[4];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        cropParams.insert({batchIndex, tmpCrop});
    }
    return APP_ERR_OK;
}

APP_ERROR DynamicAippConfig::SetPaddingParams(std::vector<int> padPrm)
{
    PaddingParams tmpPad;
    tmpPad.paddingSwitch = padPrm[0];
    tmpPad.paddingSizeTop = padPrm[1];
    tmpPad.paddingSizeBottom = padPrm[2];
    tmpPad.paddingSizeLeft = padPrm[3];
    tmpPad.paddingSizeRight = padPrm[4];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        paddingParams.insert({batchIndex, tmpPad});
    }
    return APP_ERR_OK;    
}

APP_ERROR DynamicAippConfig::SetDtcPixelMean(std::vector<int> meanPrm)
{
    DtcPixelMean tmpMean;
    tmpMean.dtcPixelMeanChn0 = meanPrm[0];
    tmpMean.dtcPixelMeanChn1 = meanPrm[1];
    tmpMean.dtcPixelMeanChn2 = meanPrm[2];
    tmpMean.dtcPixelMeanChn3 = meanPrm[3];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        dtcPixelMeanParams.insert({batchIndex, tmpMean});
    }
    return APP_ERR_OK;     
}

APP_ERROR DynamicAippConfig::SetDtcPixelMin(std::vector<float> minPrm)
{
    DtcPixelMin tmpMin;
    tmpMin.dtcPixelMinChn0 = minPrm[0];
    tmpMin.dtcPixelMinChn1 = minPrm[1];
    tmpMin.dtcPixelMinChn2 = minPrm[2];
    tmpMin.dtcPixelMinChn3 = minPrm[3];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        dtcPixelMinParams.insert({batchIndex, tmpMin});
    }
    return APP_ERR_OK; 
}

APP_ERROR DynamicAippConfig::SetPixelVarReci(std::vector<float> reciPrm)
{
    PixelVarReci tmpReci;
    tmpReci.dtcPixelVarReciChn0 = reciPrm[0];
    tmpReci.dtcPixelVarReciChn1 = reciPrm[1];
    tmpReci.dtcPixelVarReciChn2 = reciPrm[2];
    tmpReci.dtcPixelVarReciChn3 = reciPrm[3];
    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) {
        pixelVarReciParams.insert({batchIndex, tmpReci});
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