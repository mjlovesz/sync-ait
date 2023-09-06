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


#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include <map>
#include <string>
#include <mutex>
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "Base/ErrorCode/ErrorCode.h"
namespace Base {
const unsigned int DEFAULT_VALUE = 0;
struct DeviceContext {
    enum DeviceStatus {
        IDLE = 0,  // idle status
        USING      // running status
    } devStatus = IDLE;
    int32_t devId = DEFAULT_VALUE;
};

class DeviceManager {
public:
    virtual ~DeviceManager();
    static DeviceManager *GetInstance();
    // initailze all devices
    APP_ERROR InitDevices(std::string configFilePath = "");
    // get all devices count
    APP_ERROR GetDevicesCount(uint32_t& deviceCount);
    // get current running device
    APP_ERROR GetCurrentDevice(DeviceContext& device);
    APP_ERROR CreateContext(DeviceContext device, size_t& contextIndex);
    // set one device for running
    APP_ERROR SetContext(DeviceContext device, std::size_t contextIndex = 0);
    // free resources for one device
    APP_ERROR ResetDevice(DeviceContext device);
    // release all devices
    APP_ERROR DestroyDevices();
    // release specific context in a device
    APP_ERROR DestroyContext(uint32_t deviceId, std::size_t contextIndex, bool& isFinalize);
    APP_ERROR SetDeviceSimple(DeviceContext device);
    bool IsInitDevices() const;
    APP_ERROR CheckDeviceId(int32_t deviceId);
    void SetAclJsonPath(std::string aclJsonPath);
private:
    DeviceManager() = default;
    std::mutex mtx_ = {};
    std::map<long long, std::map<std::size_t, aclrtContext>> contexts_ = {};
    std::map<long long, size_t> nextContextIndex_ = {};
    uint32_t deviceCount_ = 0;
    uint32_t initCounter_ = 0;
    std::string aclJsonPath_ = "";
};
}  // namespace Base

#endif  // DEVICE_MANAGER_H
