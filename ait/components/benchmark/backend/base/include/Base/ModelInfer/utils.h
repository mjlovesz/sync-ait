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

#ifndef UTILS_H_
#define UTILS_H_
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <dirent.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <vector>

#include "Base/Log/Log.h"
#include "Base/Tensor/TensorBase/TensorBase.h"
#include "Base/ModelInfer/cnpy.h"

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

/**
* Utils
*/
class Utils {
public:
    /**
    * @brief create device buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return device buffer of file
    */
    static void* GetDeviceBufferOfFile(std::string fileName, uint32_t& fileSize);

    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return buffer of pic
    */
    static void* ReadBinFile(std::string fileName, uint32_t& fileSize);

    static void SplitString(std::string& s, std::vector<std::string>& v, char c);

    static  void SplitStringSimple(std::string str, std::vector<std::string> &out, char split1,
        char split2, char split3);

    static void SplitStringWithSemicolonsAndColons(std::string str, std::vector<std::string> &out,
        char split1, char split2);

    static  void SplitStringWithPunctuation(std::string str, std::vector<std::string> &out, char split);

    static Result SplitStingGetNameDimsMulMap(std::vector<std::string> in_dym_shape_str,
        std::map<string, int64_t> &out_namedimsmul_map);

    static int str2num(char* str);

    static std::string modelName(std::string& s);

    static std::string TimeLine();

    static std::string printCurrentTime();

    static void printHelpLetter();

    static double printDiffTime(time_t begin, time_t end);

    static double InferenceTimeAverage(double* x, int len);

    static double InferenceTimeAverageWithoutFirst(double* x, int len);

    static void ProfilerJson(bool isprof, std::map<char, std::string>& params);

    static void DumpJson(bool isdump, std::map<char, std::string>& params);

    static int ScanFiles(std::vector<std::string> &fileList, std::string inputDirectory);

    static int ToInt(std::string &str);

    static Result ReadBinFileToMemory(const std::string fileName,  char *ptr, const size_t size, size_t &offset);
    static Result FillFileContentToMemory(const std::string file, char* ptr, const size_t size, size_t &offset);

    static std::string MergeStr(std::vector<std::string>& list, const std::string& delimiter);
    static std::string GetPrefix(const std::string& outputDir, std::string filePath, const std::string& removeTail);
    static std::string RemoveSlash(const std::string& name);
    static std::string CreateDynamicShapeDims(const std::string& name, std::vector<size_t>& shapes);
    static Result TensorToNumpy(const std::string& outputFileName, Base::TensorBase& output);
    static Result TensorToBin(const std::string& outputFileName, Base::TensorBase& output);
    static Result TensorToTxt(const std::string& outputFileName, Base::TensorBase& output);
    static bool TailContain(const std::string& str, const std::string& tail);
};

#endif
