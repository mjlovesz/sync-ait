#include "binfile.h"

BinFile::BinFile() {}
BinFile::~BinFile() {}

bool BinFile::AddAttr(const std::string &name, const std::string &value)
{
    if (attrNames_.find(name) != attrNames_.end())
    {   
        std::cout << "Attr: " << name << " already exists" << std::endl;
        return false;
    }
    attrNames_.insert(name);
    attrs_.push_back({name, value});

    return true;
}

bool BinFile::Write(const std::string *filePath, const mode_t mode=BIN_FILE_MODE)
{
    
}

bool AddObject(const std::string name, const void* binaryBuffer, uint64_t binaryLen)
{

}