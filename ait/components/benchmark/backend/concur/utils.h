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

#ifndef BACKEND_UTILS_H
#define BACKEND_UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>

namespace chr = std::chrono;
using TimePointPair = std::pair<chr::steady_clock::time_point, chr::steady_clock::time_point>;
using Arguments = std::unordered_map<std::string, std::string>;

void readArgs(int argc, char *argv[], Arguments& arguments);
std::string merge_str(std::vector<std::string> list, std::string delimiter); // merge a vector of string with delimiter
std::vector<std::string> split_str(std::string input, char delimiter);
std::vector<size_t> strVecToNumVec(const std::vector<std::string>& vec);
std::vector<std::string> traversal(const char* dir); // traversal a directory return vector of filename
int createFilesList(std::vector<std::vector<std::string>>& fileList, std::string input);
std::string getPrefix(std::string filePath);
std::string removeSlash(std::string name);
std::string createDynamicShape(std::string name, std::vector<size_t> shapes);
void printTimeWall(const std::string& phase, const std::vector<TimePointPair>& timestamps);

#endif // BACKEND_UTILS_H