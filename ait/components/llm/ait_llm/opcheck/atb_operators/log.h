/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef AIT_OPCHECKER_LOG_H
#define AIT_OPCHECKER_LOG_H

#include <iostream>

static bool opcheckLogEnabled = []() -> bool {
    constexpr int logStatusOn = 1;
    const char* logStatus = std::getenv("LIB_OPCHECKER_LOG_ON");
    if (logStatus != nullptr) {
        int value = std::stoi(logStatus);
        if (value == logStatusOn) {
            return true;
        }
    }
    return false;
}();

#define AIT_OPCHECKER_COUT_LOG std::cout << "\n"
#define AIT_OPCHECKER_LOG(level) if (opcheckLogEnabled) AIT_OPCHECKER_LOG_##level
#define AIT_OPCHECKER_LOG_TRACE AIT_OPCHECKER_COUT_LOG
#define AIT_OPCHECKER_LOG_DEBUG AIT_OPCHECKER_COUT_LOG
#define AIT_OPCHECKER_LOG_INFO AIT_OPCHECKER_COUT_LOG
#define AIT_OPCHECKER_LOG_WARN AIT_OPCHECKER_COUT_LOG
#define AIT_OPCHECKER_LOG_ERROR AIT_OPCHECKER_COUT_LOG
#define AIT_OPCHECKER_LOG_FATAL AIT_OPCHECKER_COUT_LOG
#define AIT_OPCHECKER_LOG_IF(condition, level) \
    if (opcheckLogEnabled && condition) \
    AIT_OPCHECKER_LOG(level)

#endif
