/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_INFEROPPARAM_H
#define ATB_INFEROPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "atb/svector.h"

namespace atb {
namespace infer {
enum QuantType : int {
    QUANT_UNDEINFED = 0,
    QUANT_INT4, // 当前不支持
    QUANT_INT8,
    QUANT_INT16,   // 当前不支持
    QUANT_FLOAT8,  // 当前不支持
    QUANT_FLOAT16, // 当前不支持
};

enum ActivationType : int {
    ACTIVATION_UNDEFINED = 0,
    ACTIVATION_RELU,
    ACTIVATION_GELU,
    ACTIVATION_FAST_GELU,
    ACTIVATION_SWISH,
    ACTIVATION_LOG,
    ACTIVATION_SWIGLU_FORWARD,
    ACTIVATION_SWIGLU_BACKWARD,
    ACTIVATION_MAX,
};

enum CommMode : int {
    COMM_UNDEFINED = -1,
    COMM_MULTI_PROCESS,
    COMM_MULTI_THREAD,
};

struct ActivationParam {
    ActivationType activationType;
    float scale = 1.0f; // for Swish
    int32_t dim = -1; // for Swiglu
};

struct AsStridedParam {
    SVector<int64_t> size;      // size > 0
    SVector<int64_t> stride;    // stride >= 0
    SVector<int64_t> offset;    // offset >= 0
};

struct CumsumParam {
    SVector<int64_t> axes;
    bool exclusive = false;
    bool reverse = false;
};

struct GatherParam {
    int64_t axis = 0;
};

struct MatmulParam {
    bool transposeA = false; // 是否转置A矩阵
    bool transposeB = true;  // 是否转置B矩阵
};

struct MultinomialParam {
    uint32_t numSamples = 1;    // 小于等于输入tensor对应dim大小
    uint32_t randSeed = 0;
};

struct SplitParam {
    int32_t splitDim = 0;
    int32_t splitNum = 2;
};

struct ConcatParam {
    int concatDim = 0;
};

struct SliceParam {
    SVector<int64_t> offsets;
    SVector<int64_t> size;  // size >= -1
};

struct SoftmaxParam {
    SVector<int64_t> axes;  // 不为空，且连续；元素最大取值为8
};


struct TransposeParam {
    SVector<int32_t> perm;
};

struct ElewiseParam {
    enum ElewiseType : int {
        ELEWISE_UNDEFINED = 0,
        ELEWISE_CAST,
        ELEWISE_MULS,
        ELEWISE_COS,
        ELEWISE_SIN,
        ELEWISE_NEG,
        ELEWISE_QUANT,
        ELEWISE_LOGICAL_NOT,
        ELEWISE_ADD,
        ELEWISE_MUL,
        ELEWISE_REALDIV,
        ELEWISE_LOGICAL_AND,
        ELEWISE_LOGICAL_OR,
        ELEWISE_LESS,
        ELEWISE_GREATER,
        ELEWISE_SUB,
        ELEWISE_EQUAL,
        ELEWISE_QUANT_PER_CHANNEL,
        ELEWISE_DEQUANT_PER_CHANNEL,
    };

    struct QuantParam {
        float inputScale = 1.0f;
        int inputOffset = 0;
    };

    struct MulsParam {
        float varAttr = 0.0f;
    };

    ElewiseType elewiseType = ELEWISE_UNDEFINED;
    QuantParam quantParam;
    MulsParam mulsParam;

    aclDataType outTensorType = ACL_DT_UNDEFINED;
};

struct KvCacheParam {};

struct ReshapeAndCacheParam {};

struct LinearActivationParam {
    bool transposeA = false;                                  // 是否转置A矩阵
    bool transposeB = false;                                  // 是否不转置B矩阵
    ActivationType activationFuncType = ACTIVATION_FAST_GELU; // 激活函数类型
    bool hasBias = true;                                      // 是否叠加偏置
};

struct LinearActivationQuantParam {
    bool transposeA = false;                                  // 是否转置A矩阵
    bool transposeB = false;                                  // 是否不转置B矩阵
    ActivationType activationFuncType = ACTIVATION_FAST_GELU; // 激活函数类型
    bool hasBias = true; // 是否叠加偏置，该偏置为反量化偏置，当前只支持 hasBias = true
};

struct LayerNormParam {
    enum LayerNormType : int {
        LAYER_NORM_UNDEFINED = 0,
        LAYER_NORM_NORM,
        LAYER_NORM_PRENORM,
        LAYER_NORM_POSTNORM,
    };

    struct NormParam {
        QuantType quantType = QUANT_UNDEINFED;
        float epsilon = 1e-5;
        int32_t beginNormAxis = 0;
        int32_t beginParamsAxis = 0;
        float quantInputScale = 1.0f; // when quantType != QUANT_UNDEINFED, should set this
        int quantInputOffset = 0;     // when quantType != QUANT_UNDEINFED, should set this
        float quantInputAlpha = 1.0f; // when quantType != QUANT_UNDEINFED, should set this
    };

    struct PostNormParam {
        QuantType quantType = QUANT_UNDEINFED;
        float epsilon = 1e-5;
        size_t opMode = 0; // 0: high precision  1: high performance, also for quant
        float zoomScaleValue = 1.0f;
        float quantInputScale = 1.0f; // when quantType != QUANT_UNDEINFED, should set this
        int quantInputOffset = 0;     // when quantType != QUANT_UNDEINFED, should set this
        float quantInputAlpha = 1.0f; // when quantType != QUANT_UNDEINFED, should set this
    };

    LayerNormType layerType = LAYER_NORM_UNDEFINED;
    NormParam normParam;
    PostNormParam postNormParam;
};

struct RmsNormParam {
    enum RmsNormType : int {
        RMS_NORM_UNDEFINED = 0,
        RMS_NORM_NORM,
        RMS_NORM_PRENORM,
        RMS_NORM_POSTNORM,
    };

    struct NormParam {
        QuantType quantType = QUANT_UNDEINFED;
        float epsilon = 1e-5;
        float quantInputScale = 1.0f; // when quantType != QUANT_UNDEINFED, should set this
        int quantInputOffset = 0;     // when quantType != QUANT_UNDEINFED, should set this
        double layerNormEps = 1e-5;
        bool rstd = false;
    };

    struct PreNormParam {
        QuantType quantType = QUANT_UNDEINFED;
        float epsilon = 1e-5;
        float quantInputScale = 1.0f; // when quantType != QUANT_UNDEINFED, should set this
        int quantInputOffset = 0;     // when quantType != QUANT_UNDEINFED, should set this
    };

    RmsNormType layerType = RMS_NORM_UNDEFINED;
    NormParam normParam;
    PreNormParam preNormParam;
};

struct FillParam {
    bool withMask = true;
    SVector<float> value;
    SVector<int64_t> outDim;
};

struct AllGatherParam {
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    std::string backend = "hccl";
    HcclComm hcclComm = nullptr; // only effect when hcclComm is not null
    CommMode commMode = COMM_MULTI_PROCESS;
};

struct AllReduceParam {
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    std::string allReduceType = "sum";
    std::string backend = "hccl";
    HcclComm hcclComm = nullptr; // only effect when hcclComm is not null
    CommMode commMode = COMM_MULTI_PROCESS;
};

struct BroadcastParam {
    int rank = 0;
    int rankSize = 0;
    int rankRoot = 0;
    HcclComm hcclComm = nullptr; // only effect when hcclComm is not null
    CommMode commMode = COMM_MULTI_PROCESS;
    std::string backend = "hccl";
};

// matmul+add
struct LinearParam {
    bool transposeA = false; // 是否转置A矩阵
    bool transposeB = false; // 是否不转置B矩阵
    bool hasBias = true;     // 是否叠加偏置
};

struct LinearQuantParam {
    bool transposeA = false; // 是否转置A矩阵
    bool transposeB = true;  // 是否转置B矩阵
    bool hasBias = true;     // 是否叠加偏置
};

struct LinearParallelParam {
    bool transWeight = false;                 // 权重是否不需要转置
    int rank = 0;                             // 每个进程的编号
    int rankSize = 0;                         // 总的进程数
    int rankRoot = 0;                         // 主进程编号
    std::string bias = "";                    // 是否叠加偏置，为"None"时不叠加偏置
    std::string parallelType = "RowParallel"; // 权重并行方式
    std::string backend = "hccl";             // 通信后端指示
    HcclComm hcclComm = nullptr; // hccl通信域接口获取的地址指针，only effect when hcclComm is not null
    CommMode commMode = COMM_MULTI_PROCESS;
};

struct LinearSparseParam {
    bool transposeA = false; // 是否转置A矩阵
    bool transposeB = true;  // 是否转置B矩阵
    uint32_t tilingK = 1;    // 压缩参数，由外部压缩算法决定
    uint32_t tilingN = 1;    // 压缩参数，由外部压缩算法决定
};

// linear+active+linear
struct FfnParam {
    bool firstTransposeA = false;
    bool firstTransposeB = false;
    bool firstHasBias = true;
    ActivationType activationType = ACTIVATION_FAST_GELU;
    bool secondTransposeA = false;
    bool secondTransposeB = false;
    bool secondHasBias = true;
};

// linear+active+linear
struct FfnQuantParam {
    LinearParam firstLinearParam;
    ActivationType activationFuncType = ACTIVATION_FAST_GELU;
    LinearParam secondLinearParam;
    float inputScale = 1;
    int inputOffset = 0;
};

// mix.h mixType=MIX_ROPE
struct RopeParam {
    int32_t rotaryCoeff = 4;    // 2, 4, headDim / 2
    int32_t cosFormat = 0;      // 0, 1
};

// KVCache+KVCache+Muls+FlashAttention
struct SelfAttentionParam {
    int32_t headDim = 0;
    int32_t headNum = 0;
    uint32_t isTriuMask = 0;
    float qScale = 1;  // qtensor scale before qkbmm
    float qkScale = 1; // scale after qkbmm
    bool batchRunStatusEnable = false;
    int32_t kvHeadNum = 0;
    bool isEncoder = false; // encoder for pagedAttention
    enum CoderType : int {
        UNDEFINED = 0,
        ENCODER, // encoder for flashAttention
        DECODER  // decoder for flashAttention
    };
    CoderType coderType = UNDEFINED;
    bool isSupportAlibi = false;
    bool isFp32 = false; // high precision mode
    bool isClamp = false;
    float clampMin = 0;
    float clampMax = 0;
    enum MaskType : int {
        MASK_TYPE_UNDEFINED = 0,
        MASK_TYPE_NORM,
        MASK_TYPE_ALIBI
    };
    MaskType maskType = MASK_TYPE_UNDEFINED;
};

struct PagedAttentionParam { // decoder for pagedAttention
    int32_t headNum = 0;
    float qkScale = 1.0; // scale after qkbmm
    int32_t kvHeadNum = 0;
    bool isSupportAlibi = false;
    enum MaskType : int {
        UNDEFINED = 0,
        MASK_TYPE_NORM,
        MASK_TYPE_ALIBI
    };
    MaskType maskType = UNDEFINED;
    bool batchRunStatusEnable = false;
    enum QuantType: int {
        TYPE_QUANT_UNDEFINED = 0,
        TYPE_DEQUANT_FUSION
    };
    QuantType quantType = TYPE_QUANT_UNDEFINED;
    bool hasQuantOffset = false;
};

struct TransdataParam {
    enum TransdataType : int {
        UNDEFINED = 0,
        FRACTAL_NZ_TO_ND,
        ND_TO_FRACTAL_NZ
    };
    TransdataType transdataType = UNDEFINED;
    SVector<int64_t> outCrops = { 0, 0 };
};

struct WhereParam {};

struct RepeatParam {
    SVector<int64_t> multiples;
};

struct SetValueParam {
    SVector<int64_t> starts;
    SVector<int64_t> ends;
    SVector<int64_t> strides;
};

struct ReduceParam {
    enum ReduceType {
        REDUCE_UNDEFINED = 0,
        REDUCE_MAX,
        REDUCE_MIN,
    };
    ReduceType reduceType;
    SVector<int64_t> axis;  // 元素取值在输入tensor维度范围内
};

struct TopkToppSamplingParam {
    uint32_t randSeed = 0;
    uint32_t topk = 100;
};

struct PadParam {};

struct UnpadParam {};

struct SortParam {
    SVector<int32_t> num;
};
} // namespace infer
} // namespace atb
#endif
