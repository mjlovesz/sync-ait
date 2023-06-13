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

#ifndef BACKEND_CNPY_H
#define BACKEND_CNPY_H

#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <zlib.h>
#include <map>
#include <memory>
#include <stdint.h>
#include <numeric>

namespace cnpy
{
struct NpyArray {
    NpyArray(const std::vector<size_t> &shape, size_t word_size, bool fortran_order)
        : _shape(shape), _word_size(word_size), _fortran_order(fortran_order), _num_vals(1)
    {
        _num_vals = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
        _data_holder = std::make_shared<std::vector<char>>(_num_vals * _word_size);
    }
    
    NpyArray() : _shape(0), _word_size(0), _fortran_order(0), _num_vals(0) {}

    template <typename T> T *data()
    {
        return reinterpret_cast<T *>(&(*_data_holder)[0]);
    }

    template <typename T> T *data() const
    {
        return reinterpret_cast<T *>(&(*_data_holder)[0]);
    }

    template <typename T> std::vector<T> as_vec() const
    {
        const T *p = data<T>();
        return std::vector<T>(p, p + _num_vals);
    }

    size_t num_bytes() const
    {
        return _data_holder->size();
    }

    std::shared_ptr<std::vector<char>> _data_holder;
    std::vector<size_t> _shape;
    size_t _word_size;
    bool _fortran_order;
    size_t _num_vals;
};

using npz_t = std::map<std::string, NpyArray>;

char BigEndianTest();
char map_type(const std::type_info &t);
template <typename T> std::vector<char> create_npy_header(const std::vector<size_t> &shape);
void parse_npy_header(FILE *fp, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order);
void parse_npy_header(unsigned char *buffer, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order);
NpyArray npy_load(std::string fname);

template <typename T> std::vector<char> &operator += (std::vector<char> &lhs, const T rhs)
{
    for (size_t byte = 0; byte < sizeof(T); byte++) {
        char val = *(static_cast<char*>(&rhs) + byte);
        lhs.push_back(val);
    }
    return lhs;
}


// template <> std::vector<char> &operator += (std::vector<char> &lhs, const std::string rhs);
// template <> std::vector<char> &operator += (std::vector<char> &lhs, const char *rhs);

template <typename T>
void npy_save(std::string fname, const T *data, const std::vector<size_t> shape, std::string mode = "w")
{
    FILE *fp = nullptr;
    std::vector<size_t> true_data_shape;

    if (mode == "a") {
        fp = fopen(fname.c_str(), "r+b");
    }

    if (fp) {
        size_t word_size;
        bool fortran_order;
        parse_npy_header(fp, word_size, true_data_shape, fortran_order);
        assert(!fortran_order);

        if (word_size != sizeof(T)) {
            std::cout << "libnpy error: " << fname << " has word size " << word_size <<
                " but npy_save appending data sized " << sizeof(T) << "\n";
            assert(word_size == sizeof(T));
        }
        if (true_data_shape.size() != shape.size()) {
            std::cout << "libnpy error: npy_save attempting to append misdimensioned data to " << fname << "\n";
            assert(true_data_shape.size() != shape.size());
        }
        for (size_t i = 1; i < shape.size(); i++) {
            if (shape[i] != true_data_shape[i]) {
                std::cout << "libnpy error: npy_save attempting to append misshaped data to  " << fname << "\n";
                assert(shape[i] == true_data_shape[i]);
            }
        }
        true_data_shape[0] += shape[0];
    } else {
        fp = fopen(fname.c_str(), "wb");
        true_data_shape = shape;
    }

    std::vector<char> header = create_npy_header<T>(true_data_shape);
    size_t nels = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    fseek(fp, 0, SEEK_SET);
    fwrite(&header[0], sizeof(char), header.size(), fp);
    fseek(fp, 0, SEEK_END);
    fwrite(data, sizeof(T), nels, fp);
    fclose(fp);
}

template <typename T> void npy_save(std::string fname, const std::vector<T> data, std::string mode = "w")
{
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npy_save(fname, &data[0], shape, mode);
}


template <typename T> std::vector<char> create_npy_header(const std::vector<size_t> &shape)
{
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    dict+= std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); i++) {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1){
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

#endif // BACKEND_CNPY_H