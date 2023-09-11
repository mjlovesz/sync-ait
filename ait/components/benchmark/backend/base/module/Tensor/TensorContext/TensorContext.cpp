/*
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

#include "Base/Tensor/TensorContext/TensorContext.h"
#include "Base/DeviceManager/DeviceManager.h"
#include "Base/Log/Log.h"
#include "acl/acl.h"

namespace {
    const uint32_t MAX_CONTEXT_NUM = 5;
    const uint32_t MAX_QUEUE_LENGHT = 1000;
}
namespace Base {
TensorContext::TensorContext()
{
#ifdef COMPILE_PYTHON_MODULE
#endif
    if (!DeviceManager::GetInstance()->IsInitDevices()) {
        APP_ERROR ret = DeviceManager::GetInstance()->InitDevices();
        if (ret != APP_ERR_OK) {
            LogError << "DeviceManager InitDevices failed. ret=" << ret << std::endl;
            return;
        }
        InitDeviceFlag_ = true;
    }
}

APP_ERROR TensorContext::Finalize()
{
    if (InitDeviceFlag_) {
        APP_ERROR ret = DeviceManager::GetInstance()->DestroyDevices();
        if (ret != APP_ERR_OK) {
            LogError << "DeviceManager DestroyDevices failed. ret=" << ret << std::endl;
            return ret;
        }
        InitDeviceFlag_ = false;
    }
    return APP_ERR_OK;
}

TensorContext::~TensorContext()
{
    Finalize();
}

APP_ERROR TensorContext::CreateContext(const uint32_t &deviceId, size_t& contextIndex)
{
    DeviceContext device = {};
    device.devId = deviceId;
    APP_ERROR ret = DeviceManager::GetInstance()->CreateContext(device, contextIndex);
    if (ret != APP_ERR_OK) {
        LogError << "CreateContext failed. ret=" << ret << std::endl;
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR TensorContext::DestroyContext(const uint32_t &deviceId, const size_t& contextIndex)
{
    APP_ERROR ret = DeviceManager::GetInstance()->DestroyContext(deviceId, contextIndex);
    if (ret != APP_ERR_OK) {
        LogError << "DestroyContext failed. ret=" << ret << std::endl;
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR TensorContext::SetContext(const uint32_t &deviceId, const size_t contextIndex)
{
    DeviceContext device = {};
    device.devId = deviceId;
    APP_ERROR ret = DeviceManager::GetInstance()->SetContext(device, contextIndex);
    if (ret != APP_ERR_OK) {
        LogError << "SetContext failed. ret=" << ret << std::endl;
        return ret;
    }
    return APP_ERR_OK;
}

std::shared_ptr<TensorContext> TensorContext::GetInstance()
{
    static std::shared_ptr<TensorContext> tensorContext = std::make_shared<TensorContext>();
    return tensorContext;
}
}
