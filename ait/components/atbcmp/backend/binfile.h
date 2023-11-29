#ifndef BINFILE_H
#define BINFILE_H

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>


const std::string ATTR_VERSION = "$Version";
const std::string ATTR_END = "$END";
const std::string ATTR_OBJECT_LENGTH = "$Object.Length";
const std::string ATTR_OBJECT_COUNT = "$Object.Count";
const std::string ATTR_OBJECT_PREFIX = "$Object.";
constexpr mode_t BIN_FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP;

class BinFile {
struct Binary {
    uint64_t offset = 0;
    uint64_t length = 0;
};
public:
    BinFile();
    ~BinFile();

    bool AddAttr(const std::string &name, const std::string &value);
    bool Write(const std::string *filePath, const mode_t mode=BIN_FILE_MODE);
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