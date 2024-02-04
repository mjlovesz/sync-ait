/**
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
#ifndef COMMON_FILESYSTEM_FILESYSTEM_H
#define COMMON_FILESYSTEM_FILESYSTEM_H
#include <string>
#include <vector>
#include <sys/stat.h>
namespace AsdOps {
class FileSystem {
public:
    static bool Exists(const std::string &path);
    static std::string DirName(const std::string &path);
    static bool Makedirs(const std::string &dirPath, const mode_t mode);
};
} // namespace AsdOps
#endif