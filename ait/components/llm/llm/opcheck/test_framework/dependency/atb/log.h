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

#ifndef ATB_LOG_H
#define ATB_LOG_H

#include <iostream>

#define ATB_COUT_LOG std::cout << "\n"
#define ATB_LOG(level) ATB_LOG_##level
#define ATB_LOG_TRACE ATB_COUT_LOG
#define ATB_LOG_DEBUG ATB_COUT_LOG
#define ATB_LOG_INFO ATB_COUT_LOG
#define ATB_LOG_WARN ATB_COUT_LOG
#define ATB_LOG_ERROR ATB_COUT_LOG
#define ATB_LOG_FATAL ATB_COUT_LOG
#define ATB_LOG_IF(condition, level) \
    if (condition)                   \
    ATB_LOG(level)

#endif
