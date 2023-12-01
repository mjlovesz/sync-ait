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


#ifndef BINFILE_H
#define BINFILE_H

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <set>
#include <map>
#include <sstream>
#include <fstream>

const std::string ATTR_VERSION = "$Version";
const std::string ATTR_END = "$END";
const std::string ATTR_OBJECT_LENGTH = "$Object.Length";
const std::string ATTR_OBJECT_COUNT = "$Object.Count";
const std::string ATTR_OBJECT_PREFIX = "$Object.";
const std::string END_VALUE = "1";

constexpr mode_t BIN_FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP;

constexpr uint64_t MAX_SINGLE_MEMCPY_SIZE = 1073741824;
class BinFile {
struct Binary {
    uint64_t offset = 0;
    uint64_t length = 0;
};
public:
    BinFile();
    ~BinFile();

    bool AddAttr(const std::string &name, const std::string &value);
    bool Write(const std::string &filePath, const mode_t mode=BIN_FILE_MODE);
    bool WriteAttr(std::ofstream &outputFile, const std::string &filePath, const std::string &value);
    bool AddObject(const std::string name, const void* binaryBuffer, uint64_t binaryLen);

private:
    std::string version_ = "1.0";
    std::set<std::string> attrNames_;
    std::vector<std::pair<std::string, std::string>> attrs_;

    std::set<std::string> binaryNames_;
    std::vector<std::pair<std::string, Binary>> binaries_;
    std::vector<char> binariesBuffer_;
};
#endif