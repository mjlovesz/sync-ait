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

#ifndef _CNPY_H
#define _CNPY_H

#include <string>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <zlib.h>
#include <map>
#include <memory>
#include <stdint.h>
#include <numeric>
#include <cstdlib>
#include <ctime>

#include "Base/Log/Log.h"

namespace cnpy
{
struct NpyArray {
    NpyArray(const std::vector<size_t> &shape, size_t wordSize, bool fortranOrder)
        : shape(shape), wordSize(wordSize), fortranOrder(fortranOrder), numVals(1)
    {
        numVals = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        dataHolder = std::make_shared<std::vector<char>>(numVals * wordSize);
    }

    NpyArray() : shape(0), wordSize(0), fortranOrder(0), numVals(0) {}

    template <typename T> T *Data()
    {
        return reinterpret_cast<T *>(&(*dataHolder)[0]);
    }

    template <typename T> T *Data() const
    {
        return reinterpret_cast<T *>(&(*dataHolder)[0]);
    }

    template <typename T> std::vector<T> AsVec() const
    {
        const T *p = Data<T>();
        return std::vector<T>(p, p + numVals);
    }

    size_t NumBytes() const
    {
        return dataHolder->size();
    }

    std::shared_ptr<std::vector<char>> dataHolder;
    std::vector<size_t> shape;
    size_t wordSize;
    bool fortranOrder;
    size_t numVals;
};

union DataUnion {
    uint8_t value;
    char bytes;
};

using npz_t = std::map<std::string, NpyArray>;

char BigEndianTest();
char MapType(const std::type_info &t);
template <typename T> std::vector<char> CreateNpyHeader(const std::vector<size_t> &shape);
void ParseNpyHeader(FILE *fp, size_t &wordSize, std::vector<size_t> &shape, bool &fortranOrder);
void ParseNpyHeader(unsigned char *buffer, size_t &wordSize, std::vector<size_t> &shape, bool &fortranOrder);
NpyArray NpyLoad(std::string fname);
NpyArray BinLoad(std::string fname);

template <typename T> std::vector<char> &operator += (std::vector<char> &lhs, const T rhs)
{
    for (size_t byte = 0; byte < sizeof(T); byte++) {
        char val = *((char*)(&rhs) + byte);
        lhs.push_back(val);
    }
    return lhs;
}

template <> std::vector<char> &operator += (std::vector<char> &lhs, const std::string rhs);
template <> std::vector<char> &operator += (std::vector<char> &lhs, const char *rhs);

template <typename T>
void NpySave(std::string fname, const T *data, const std::vector<size_t> shape, std::string mode = "w")
{
    FILE *fp = nullptr;
    std::vector<size_t> trueDataShape;

    if (mode == "a") {
        fp = fopen(fname.c_str(), "r+b");
    }

    if (fp) {
        size_t wordSize;
        bool fortranOrder;
        ParseNpyHeader(fp, wordSize, trueDataShape, fortranOrder);
        if (fortranOrder) {
            throw std::runtime_error("NpySave: fortranOrder wrong");
        }

        if (wordSize != sizeof(T)) {
            ERROR_LOG("libnpy error: %s has word size %zu but NpySave appending data sized %zu\n",
                      fname.c_str(), wordSize, sizeof(T));
            throw std::runtime_error("NpySave: wordSize not matching");
        }
        if (trueDataShape.size() != shape.size()) {
            ERROR_LOG("libnpy error: NpySave attempting to append misdimensioned data to %s\n", fname.c_str());
            throw std::runtime_error("NpySave: dimension not matching");
        }
        for (size_t i = 1; i < shape.size(); i++) {
            if (shape[i] != trueDataShape[i]) {
                ERROR_LOG("libnpy error: NpySave attempting to append misshaped data to %s", fname.c_str());
                throw std::runtime_error("NpySave: shape not matching");
            }
        }
        trueDataShape[0] += shape[0];
    } else {
        fp = fopen(fname.c_str(), "wb");
        trueDataShape = shape;
    }

    std::vector<char> header = CreateNpyHeader<T>(trueDataShape);
    size_t nels = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    fseek(fp, 0, SEEK_SET);
    fwrite(&header[0], sizeof(char), header.size(), fp);
    fseek(fp, 0, SEEK_END);
    fwrite(data, sizeof(T), nels, fp);
    fclose(fp);
}

template <typename T> void NpySave(std::string fname, const std::vector<T> data, std::string mode = "w")
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    NpySave(fname, &data[0], shape, mode);
}


template <typename T> std::vector<char> CreateNpyHeader(const std::vector<size_t> &shape)
{
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += MapType(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortranOrder': False, 'shape': (";
    dict+= std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); i++) {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1) {
        dict += ",";
    }
    dict += "), }";
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += static_cast<char>(0x93);
    header += "NUMPY";
    header += static_cast<char>(0x01);
    header += static_cast<char>(0x00);
    header += static_cast<uint16_t>(dict.size());
    header.insert(header.end(), dict.begin(), dict.end());

    return header;
}
} // namespace cnpy

#endif // _CNPY_H