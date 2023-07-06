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

#include <typeinfo>
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <regex>

#include "Base/ModelInfer/cnpy.h"

#define UPPER_BOUND_FILE 1 << 30

char cnpy::BigEndianTest()
{
    int x = 1;
    return (((char*)(&x))[0]) ? '<' : '>';
}

char cnpy::MapType(const std::type_info &t)
{
    if (t == typeid(float) || t == typeid(double) || t == typeid(long double)) {
        return 'f';
    }

    if (t == typeid(int) || t == typeid(char) || t == typeid(short) || t == typeid(long) || t == typeid(long long)) {
        return 'i';
    }

    if (t == typeid(unsigned char) || t == typeid(unsigned short) || t == typeid(unsigned long) ||
        t == typeid(unsigned long long) || t == typeid(unsigned int)) {
        return 'u';
    }

    if (t == typeid(bool)) {
        return 'b';
    }

    if (t == typeid(std::complex<float>) || t == typeid(std::complex<double>) ||
        t == typeid(std::complex<long double>)) {
        return 'c';
    }

    return '?';
}

template <> std::vector<char> &cnpy::operator += (std::vector<char> &lhs, const std::string rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template <> std::vector<char> &cnpy::operator += (std::vector<char> &lhs, const char *rhs)
{
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

void cnpy::ParseNpyHeader(unsigned char *buffer, size_t &wordSize, std::vector<size_t> &shape, bool &fortranOrder)
{
    uint16_t headerLen = *reinterpret_cast<uint16_t *>(buffer + 8);            // 8 means offset of headerLen
    std::string header(reinterpret_cast<char *>(buffer + 9), headerLen);       // 9 means offser of header

    size_t loc1, loc2;

    loc1 = header.find("fortran_order") + 16; // 16 means offset
    fortranOrder = (header.substr(loc1, 4) == "True" ? true : false); // 4 means length of "True"

    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex numRegex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();
    std::string strShape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(strShape, sm, numRegex)) {
        shape.push_back(std::stoi(sm[0].str()));
        strShape = sm.suffix().str();
    }

    loc1 = header.find("descr") + 9; // 9 means offset
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    if (!littleEndian) {
        throw std::runtime_error("ParseNpyHeader: should be little endian.");
    }

    std::string strWs = header.substr(loc1 + 2);
    loc2 = strWs.find("'");
    wordSize = atoi(strWs.substr(0, loc2).c_str());
}

void cnpy::ParseNpyHeader(FILE *fp, size_t &wordSize, std::vector<size_t> &shape, bool &fortranOrder)
{
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11) { // 11 means buffer size
        throw std::runtime_error("ParseNpyHeader: failed fread");
    }
    std::string header = fgets(buffer, 256, fp);
    if (header[header.size() - 1] != '\n') {
        throw std::runtime_error("ParseNpyHeader: the ending of header should be \n.");
    }

    size_t loc1, loc2;

    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos) {
        throw std::runtime_error("ParseNpyHeader: failed to find header keyword : 'fortranOrder'");
    }
    loc1 += 16; // 16 menas offset
    fortranOrder = (header.substr(loc1, 4) == "True" ? true :false); // 4 means length of "True"

    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos) {
        throw std::runtime_error("ParseNpyHeader: failed to find header keyword: '(' or ')'");
    }
    std::regex numRegex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string strShape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(strShape, sm, numRegex)) {
        shape.push_back(std::stoi(sm[0].str()));
        strShape = sm.suffix().str();
    }

    loc1 = header.find("descr");
    if (loc1 == std::string::npos) {
        throw std::runtime_error("ParseNpyHeader: failed to find header keyword : 'descr'");
    }
    loc1 += 9; // 9 menas offset
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    if (!littleEndian) {
        throw std::runtime_error("ParseNpyHeader: should be little endian.");
    }

    std::string strWs = header.substr(loc1 + 2);
    loc2 = strWs.find("'");
    wordSize = atoi(strWs.substr(0, loc2).c_str());
}

cnpy::NpyArray LoadNpyFile(FILE *fp)
{
    std::vector<size_t> shape;
    size_t wordSize;
    bool fortranOrder;
    cnpy::ParseNpyHeader(fp, wordSize, shape, fortranOrder);
    if (wordSize > UPPER_BOUND_FILE) {
        throw std::runtime_error("LoadNpyFile: file size greater than upper bound");
    }

    cnpy::NpyArray arr(shape, wordSize, fortranOrder);
    size_t nread = fread(arr.Data<char>(), 1, arr.NumBytes(), fp);
    if (nread != arr.NumBytes()) {
        throw std::runtime_error("LoadNpyFile: failed fread");
    }
    return arr;
}

cnpy::NpyArray cnpy::NpyLoad(std::string fname)
{
    FILE *fp = fopen(fname.c_str(), "rb");

    if (!fp) {
        throw std::runtime_error("NpyLoad: Unable to open file" + fname);
    }
    
    NpyArray arr = LoadNpyFile(fp);

    fclose(fp);
    return arr;
}

