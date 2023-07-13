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
#include "AscendIE.h"
#include <iostream>
#include <fstream>

using namespace AscendIE;

int main(int argc, char** argv)
{
    const int inputIndex = 4;
    std::string onnxModelPath = argv[1];
    std::string outputPath = argv[2];
    std::string socVersion = argv[3];
    if (argc > inputIndex) {
        std::cout << "For AIE model convert, only three parameters: model, output and soc_version are needed." << std::endl;
    }

    Builder* builder = Builder::CreateInferBuilder(socVersion);

    auto network = builder->CreateNetwork();

    std::string moddelPath(onnxModelPath);

    OnnxModelParser parser;

    bool ret = parser.Parse(network, onnxModelPath.c_str());

    BuilderConfig config;

    config.SetFlag(BuilderFlag::FP16);

    auto modelData = builder->BuildModel(network, config);

    std::ofstream fout(outputPath, std::ios::binary);

    fout.write((char*)modelData.data.get(), modelData.size);

    fout.close();

    std::cout << "AIE Model Convert" << std::endl;

    delete builder;
}
