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

#include "atb_speed_probe.h"

namespace atb_speed {
bool atb_speed::SpeedProbe::IsSaveTopoInfo()
{
    const char* saveTopoInfo = std::getenv("ATB_SAVE_TOPO_INFO");
    if (saveTopoInfo == nullptr) {
        return false;
    }
    
    int value = std::stoi(saveTopoInfo);
    if (value == 1) {
        return true;
    }
    return false;
}

void atb_speed::SpeedProbe::SaveTopoInfo(const std::string &modelJson, const std::string &fileName)
{
    const char* outputDir = std::getenv("ATB_OUTPUT_DIR");
    std::string outDir = outputDir != nullptr ? outputDir : "./";

    std::string outPath = outDir + fileName;
    std::ofstream outfile(outPath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile << modelJson << std::endl;
        outfile.close();
        std::cout << "Model topo info written to file successfully! File name:" << outPath << std::endl;
    } else {
        std::cout << "Unable to open file!" << std::endl;
    }
    return;
}

} // namespace atb_speed