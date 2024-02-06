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
#include "filesystem.h"

#include <cstring>
#include <cstdlib>
#include <climits>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

namespace AsdOps {
constexpr size_t MAX_PATH_LEN = 256;

bool FileSystem::Exists(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) < 0) {
        return false;
    }
    return true;
}

std::string FileSystem::DirName(const std::string &path)
{
    if (path.size() < 1) {return "";}
    int32_t idx = static_cast<int32_t>(path.size() - 1);
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    std::string sub = path.substr(0, static_cast<uint32_t>(idx));
    const char *str = strrchr(sub.c_str(), '/');
    if (str == nullptr) {
        return ".";
    }
    idx = str - sub.c_str() - 1;
    while (idx >= 0 && path[idx] == '/') {
        idx--;
    }
    if (idx < 0) {
        return "/";
    }
    return path.substr(0, static_cast<uint32_t>(idx) + 1);
}

static bool CheckNameValid(const std::string &name)
{
    // 1. check the name not empty
    if (name.empty()) {
        std::cout << "The name is empty" << std::endl;
        return false;
    }

    // 2. check the length of the name
    const size_t maxNameLen = 256U;
    if (name.size() > maxNameLen) {
        std::cout << "The length of the name " << name.size() << " > " << maxNameLen << std::endl;
        return false;
    }

    // 3. check the name has invalid characters
    const std::vector<char> invalidChars = { '\n', '\f', '\r', '\v', '\t', '\b', '\u007f', ' ', '$'};
    for (auto c : invalidChars) {
        if (name.find(c) != std::string::npos) {
            std::cout << "Name contains invalid characters: " << c << std::endl;
            return false;
        }
    }
    return true;
}

static bool MakeDir(const std::string &dirPath, int mode)
{
    if (!CheckNameValid(dirPath)) {
        return false;
    }
    int ret = mkdir(dirPath.c_str(), mode);
    return ret == 0;
}

bool FileSystem::Makedirs(const std::string &dirPath, const mode_t mode)
{
    uint32_t offset = 0;
    uint32_t pathLen = dirPath.size();
    do {
        const char *str = strchr(dirPath.c_str() + offset, '/');
        if (str == nullptr) {
            std::cout << "using path.size to initialize offset." << std::endl;
        }
        offset = (str == nullptr) ? pathLen : str - dirPath.c_str() + 1;
        std::string curPath = dirPath.substr(0, offset);
        if (!Exists(curPath)) {
            if (!MakeDir(curPath, mode)) {
                return false;
            }
        }
    } while (offset != pathLen);
    return true;
}
} // namespace AsdOps