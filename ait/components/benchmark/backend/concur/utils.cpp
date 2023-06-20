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

#include <iostream>
#include <sstream>
#include <string>
#include <numeric>
#include <algorithm>
#include <dirent.h>
#include <chrono>
#include <ctime>
#include <iomanip>

#include "utils.h"


void ReadArgs(int argc, char *argv[], Arguments& arguments)
{
    for (int i = 1; i < argc; ++i) {
        auto valuePtr = strchr(argv[i], '=');
        if (valuePtr) {
            std::string value { valuePtr + 1 };
            std::string key { argv[i], (long unsigned int)(valuePtr - argv[i]) };
            if (arguments.find(key) != arguments.end()) {
                arguments[key] = value;
            } else {
                ERROR_LOG("%s is not a parameter\n", key.c_str());
            }
        } else {
            ERROR_LOG("pass the parameter in form of key=value\n");
        }
    }
}


std::string MergeStr(std::vector<std::string> list, std::string delimiter)
{
    auto res = std::accumulate(list.begin(), list.end(), std::string(),
    [=](const std::string& a, const std::string& b) -> std::string {
        return a + (a.length() > 0 ? delimiter : "") + b; });
    return res;
}

std::vector<std::string> SplitStr(std::string input, char delimiter)
{
    std::stringstream inStream(input);
    std::vector<std::string> res;

    while (inStream.good()) {
        std::string subStr;
        getline(inStream, subStr, delimiter);
        res.emplace_back(subStr);
    }
    return res;
}

std::vector<size_t> StrVecToNumVec(const std::vector<std::string>& strVec)
{
    std::vector<size_t> res;
    for (auto &elem : strVec) {
        res.emplace_back(stoi(elem));
    }
    return res;
}

std::vector<std::string> Traversal(const char* dir)
{
    DIR *dir_ptr;
    struct dirent *diread;
    std::vector<std::string> filenames;
    if ((dir_ptr = opendir(dir)) != nullptr) {
        while ((diread = readdir(dir_ptr)) != nullptr) {
            if (diread->d_type == DT_REG) {
                filenames.emplace_back(std::string(dir) + diread->d_name);
            }
        }
        closedir(dir_ptr);
    }
    std::sort(filenames.begin(), filenames.end());
    return filenames;
}

// input need to be dir1,dir2,dir3,...
Result CreateFilesList(std::vector<std::vector<std::string>>& fileList, std::string input)
{
    std::vector<std::vector<std::string>> directorys;

    for (auto &dir : SplitStr(input, ',')) {
        if (dir.back() != '/') {
            dir.push_back('/');
        }
        directorys.emplace_back(std::move(Traversal(dir.c_str())));
    }
    // check whether number of files in each directory is the same
    size_t n = directorys[0].size();
    for (auto &directory : directorys) {
        if (directory.size() != n) {
            return FAILED;
        }
    }

    for (size_t i = 0; i < n; i++) {
        std::vector<std::string> combine {};
        for (auto &directory : directorys) {
            combine.emplace_back(directory[i]);
        }
        fileList.emplace_back(std::move(combine));
    }
    return SUCCESS;
}

std::string GetPrefix(std::string filePath)
{
    std::stringstream inStream(filePath);
    std::string res {};
    while (inStream.good()) {
        std::string subStr;
        getline(inStream, subStr, '/');
        if (subStr == "") {
            continue;
        }
        res = subStr;
    }
    return res;
}

std::string RemoveSlash(std::string name)
{
    std::string res;
    for (auto &elem: name) {
        if (elem != '/') {
            res.push_back(elem);
        }
    }
    return res;
}

std::string CreateDynamicShape(std::string name, std::vector<size_t> shapes)
{
    std::vector<std::string> shapeStr {};
    for (auto &shape : shapes) {
        shapeStr.emplace_back(std::to_string(shape));
    }
    auto res = MergeStr(shapeStr, ",");
    return name + ":" + res;
}

void PrintTimeWall(const std::string& phase, const std::vector<TimePointPair>& timeStamps)
{
    if (timeStamps.empty()) {
        return;
    }
    auto [minIt, maxIt] = std::minmax_element(timeStamps.begin(), timeStamps.end(),
                                              [](auto tp1, auto tp2)
                                              {return tp1.second - tp1.first < tp2.second - tp2.first;});

    auto total = std::accumulate(timeStamps.begin(), timeStamps.end(), 0,
                                 [](auto init, auto tpp)
                                 {return init + chr::duration_cast<chr::microseconds>(tpp.second - tpp.first).count();});

    float division = 1000.0; // microsecond to millisecond
    auto avg = total/timeStamps.size();
    INFO_LOG("%s avg: %fms. min: %fms. max: %f ms.\n", phase.c_str(), avg/division,
             chr::duration_cast<chr::microseconds>(minIt->second - minIt->first).count()/division,
             chr::duration_cast<chr::microseconds>(maxIt->second - maxIt->first).count()/division);
}

std::string RemoveTail(std::string src, const std::string tail)
{
    if (src.size() >= tail.size() && src.compare(src.size() - tail.size(), tail.size(), tail) == 0) {
        src.erase(src.size() - tail.size());
    }
    return src;
}

std::string GetCurrentTime()
{
    std::time_t now = std::time(nullptr);
    std::tm time = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&time, "%Y_%m_%d-%H_%M_%S");
    return oss.str();
}