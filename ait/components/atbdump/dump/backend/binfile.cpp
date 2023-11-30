#include "binfile.h"

#define x first
#define y second

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