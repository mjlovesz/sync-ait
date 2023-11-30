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


#include "binfile.h"

#define x first
#define y second


static bool directoryExists(const std::string &path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}


BinFile::BinFile() {}
BinFile::~BinFile() {}

bool BinFile::AddAttr(const std::string &name, const std::string &value)
{   // **********todo name检查*************
    if (attrNames_.find(name) != attrNames_.end())
    {   
        std::cout << "Attr: " << name << " already exists" << std::endl;
        return false;
    }
    attrNames_.insert(name);
    attrs_.push_back({name, value});

    return true;
}

bool BinFile::Write(const std::string &filePath, const mode_t mode)
{
    // 先写头
    // 先写version、count、length
    // 写format dtype dims
    // 再写data
    // 再写end
    size_t found = filePath.find_last_of("/");
    std::string directory = filePath.substr(0, found);

    // 检查目录是否存在，如果不存在则创建目录和文件
    if (!directoryExists(directory)) {
        int status = mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status == 0) {

            std::cout << "directory created: " << directory << std::endl;
        
        } else {
            std::cout << "cannot create directory: " << directory << std::endl;
            return false;
        }
    }

    std::ofstream outputFile(filePath, std::ios::app);
    if (!outputFile.is_open())
    {
        std::cout << "File to write can't open : " << filePath << std::endl;
    }

    bool ret = WriteAttr(outputFile, ATTR_VERSION, version_);
    ret = WriteAttr(outputFile, ATTR_OBJECT_COUNT, std::to_string(binaries_.size()));
    ret = WriteAttr(outputFile, ATTR_OBJECT_LENGTH, std::to_string(binariesBuffer_.size()));
    
    for (const auto &attrIt : attrs_)
    {
        ret = WriteAttr(outputFile, attrIt.x, attrIt.y);
    }

    for (const auto &objIt : binaries_)
    {
        ret = WriteAttr(outputFile, ATTR_OBJECT_PREFIX + objIt.x, std::to_string(objIt.y.offset) + "," + std::to_string(objIt.y.length));
    }

    ret = WriteAttr(outputFile, ATTR_END, "1");

    if (binariesBuffer_.size() > 0)
    {
        outputFile.write(binariesBuffer_.data(), binariesBuffer_.size());
    }
    return true;
}

bool BinFile::AddObject(const std::string name, const void* binaryBuffer, uint64_t binaryLen)
{
    // todo 检查name
    if (binaryBuffer == nullptr)
    {
        std::cout << "binary buffer size is none" << std::endl;
        return false;
    }
    size_t needLen = binariesBuffer_.size() + binaryLen;
    

    // todo 超出maxfilesize

    if (binaryNames_.find(name) != binaryNames_.end())
    {
        return false;
    }

    binaryNames_.insert(name);

    size_t currentLen = binariesBuffer_.size();
    BinFile::Binary binary;
    binary.offset = currentLen;
    binary.length = binaryLen;
    binaries_.push_back({name, binary});
    binariesBuffer_.resize(needLen);

    uint64_t offset = 0;
    uint64_t copyLen = binaryLen;
    while (copyLen > 0)
    {
        uint64_t curCopySize = copyLen > MAX_SINGLE_MEMCPY_SIZE ? MAX_SINGLE_MEMCPY_SIZE : copyLen;
        auto ret = memcpy(binariesBuffer_.data() + currentLen + offset, binariesBuffer_.size() - currentLen - offset,
                            static_cast<uint8_t*>(binaryBuffer) + offset, curCopySize);
        offset += curCopySize;
        copyLen -= curCopySize;
    }
    return true;
}

bool BinFile::WriteAttr(std::ofstream &outputFile, const std::string &name, const std::string &value)
{
    std::string line = name + "=" + value + "\n";
    outputFile << line;
    return true;
}