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

#ifndef SESSION_OPTIONS_H
#define SESSION_OPTIONS_H

#include "Base/ModelInfer/utils.h"

namespace Base {
class SessionOptions {
public:
    int log_level = LOG_INFO_LEVEL;
    int loop = 1;
    std::string aclJsonPath = "";
};
}
#endif