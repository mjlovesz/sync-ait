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

#ifndef INFER_OPTIONS_H
#define INFER_OPTIONS_H

#include <string>
#include <vector>

namespace Base {
class InferOptions {
public:
    std::string outputDir = "";
    bool autoDymShape = false;
    bool autoDymDims = false;
    std::string outFmt = "";
    bool pureInferMode = false;
    std::vector<std::string> outputNames;
    std::vector<std::vector<std::vector<std::size_t>>> shapesList;
};
}
#endif