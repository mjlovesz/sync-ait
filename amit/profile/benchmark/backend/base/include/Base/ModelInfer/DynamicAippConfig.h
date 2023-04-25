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
#ifndef DYNAMIC_AIPP_CONFIG_H
#define DYNAMIC_AIPP_CONFIG_H
#include <string>
#include <unordered_map>
#include <vector>
#include "Base/ErrorCode/ErrorCode.h"



namespace Base {

struct CropParams {
    int8_t cropSwitch;
    int32_t loadStartPosW;
    int32_t loadStartPosH;
    int32_t cropSizeW;
    int32_t cropSizeH;
};

struct PaddingParams {
    int8_t paddingSwitch;
    int32_t paddingSizeTop;
    int32_t paddingSizeBottom;
    int32_t paddingSizeLeft;
    int32_t paddingSizeRight;
};

/* 310,310P,910都不支持
struct ScfParams {
    int8_t scfSwitch;
    int32_t scfInputSizeW;
    int32_t scfInputSizeH;
    int32_t scfOutputSizeW;
    int32_t scfOutputSizeH;
}
*/

struct ScfParams {
    int8_t scfSwitch;
    int32_t scfInputSizeW;
    int32_t scfInputSizeH;
    int32_t scfOutputSizeW;
    int32_t scfOutputSizeH;
};

struct CscParams {
    int8_t cscSwitch;
    int32_t cscMatrixR0C0;
    int32_t cscMatrixR0C1;
    int32_t cscMatrixR0C2;
    int32_t cscMatrixR1C0;
    int32_t cscMatrixR1C1;
    int32_t cscMatrixR1C2;
    int32_t cscMatrixR2C0;
    int32_t cscMatrixR2C1;
    int32_t cscMatrixR2C2;
    int32_t cscOutputBias0;
    int32_t cscOutputBias1;
    int32_t cscOutputBias2;
    int32_t cscInputBias0;
    int32_t cscInputBias1;
    int32_t cscInputBias2;
};

struct DtcPixelMean {
    int16_t dtcPixelMeanChn0;
    int16_t dtcPixelMeanChn1;
    int16_t dtcPixelMeanChn2;
    int16_t dtcPixelMeanChn3;
};

struct DtcPixelMin {
    float dtcPixelMinChn0;
    float dtcPixelMinChn1;
    float dtcPixelMinChn2;
    float dtcPixelMinChn3;
};

struct PixelVarReci {
    float dtcPixelVarReciChn0;
    float dtcPixelVarReciChn1;
    float dtcPixelVarReciChn2;
    float dtcPixelVarReciChn3;
};

class DynamicAippConfig {
private:
    //--------动态AIPP必填参数--------
    std::string inputFormat; // 原始输入图像的格式
    int32_t srcImageSizeW; // 原始图片尺寸
    int32_t srcImageSizeH; // 原始图片尺寸
    //--------动态AIPP必填参数--------
  
    int8_t rbuvSwapSwitch; // 是否交换R通道与B通道、或者是否交换U通道与V通道
    int8_t axSwapSwitch; // RGBA->ARGB或者YUVA->AYUV的交换开关
    
    CscParams cscParams; // CSC色域转换相关的参数

    //----------多个不同 batchIndex需要设置,map的key 为batchIndex -------- 
    std::unordered_map<uint64_t, CropParams> cropParams = {}; // 抠图相关参数
    std::unordered_map<uint64_t, PaddingParams> paddingParams = {}; // 补边相关参数
    std::unordered_map<uint64_t, DtcPixelMean> dtcPixelMeanParams = {}; // 通道的均值
    std::unordered_map<uint64_t, DtcPixelMin> dtcPixelMinParams = {}; // 通道的最小值
    std::unordered_map<uint64_t, PixelVarReci> pixelVarReciParams = {};// 通道的方差
    // 310,310P,910都不支持 scfParams

    //----------多个不同 batchIndex需要设置-------- 

public:
    DynamicAippConfig();
    ~DynamicAippConfig();

    APP_ERROR SetMaxBatchSize(uint64_t maxBsParams);
    APP_ERROR SetInputFormat(std::string iptFmt);
    APP_ERROR SetSrcImageSize(std::vector<int> srcImageSize);
    APP_ERROR SetRbuvSwapSwitch(int rsSwitch);
    APP_ERROR SetAxSwapSwitch(int asSwitch);
    APP_ERROR SetCscParams(std::vector<int> cscParams);
    APP_ERROR SetCropParams(std::vector<int> cropParams);
    APP_ERROR SetPaddingParams(std::vector<int> padParams);
    APP_ERROR SetDtcPixelMean(std::vector<int> meanParams);
    APP_ERROR SetDtcPixelMin(std::vector<float> minParams);
    APP_ERROR SetPixelVarReci(std::vector<float> reciParams);
    
    uint64_t GetMaxBatchSize();
    std::string GetInputFormat();
    int32_t GetSrcImageSizeW();
    int32_t GetSrcImageSizeH();
    int8_t GetRbuvSwapSwitch();
    int8_t GetAxSwapSwitch();
    CscParams GetCscParams();
    std::unordered_map<uint64_t, CropParams> GetCropParams();
    std::unordered_map<uint64_t, PaddingParams> GetPaddingParams();
    std::unordered_map<uint64_t, DtcPixelMean> GetDtcPixelMean();
    std::unordered_map<uint64_t, DtcPixelMin> GetDtcPixelMin();
    std::unordered_map<uint64_t, PixelVarReci> GetPixelVarReci();

    bool IsActivated();
    bool ModelIsLegal();
    void ActivateConfig();
    void ActivateModel();
private:
    uint64_t maxBatchSize;
    bool isActivated; // --aipp_config文件内容读取成功
    bool modelOK; // 模型只有一个动态aipp输入

};

}  // namespace Base
#endif