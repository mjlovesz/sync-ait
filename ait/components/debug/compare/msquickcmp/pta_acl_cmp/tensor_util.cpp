#include "acltransformer/utils/tensor_util.h"

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <thread>
#include <unistd.h>

#include <asdops/utils/binfile/binfile.h>
#include <asdops/utils/log/log.h>
#include <asdops/utils/rt/rt.h>
#include <asdops/utils/filesystem/filesystem.h>


void InitialPathTable(std::unordered_set<std::string> &pathTable) {
    const char* envValue = std::getenv("AIT_CMP_TASK_PID");
    if (envValue) {
        std::string aitCmpTaskPid = envValue;
        std::string fileName = "/tmp/" + aitCmpTaskPid + "/ait_compare_acl_map.txt";

        std::ifstream fileContent(fileName);
        if (fileContent.is_open()) {
            std::string filePath;
            while (std::getline(fileContent, filePath)) {
                pathTable.insert(filePath);
            }
            fileContent.close();
        }
    }
}


std::unordered_set<std::string> &findTable() {
    static std::unordered_set<std::string> pathTable;
    if (pathTable.empty()) {
        InitialPathTable(pathTable);
    }
    return pathTable;
}


void AclTransformer::TensorUtil::SaveTensor(const AsdOps::Tensor &tensor, const std::string &filePath) {
    std::unordered_set<std::string> &copyTable = findTable();
    std::ostringstream ss;
    ss << std::this_thread::get_id();
    std::string basePath = std::string(std::getenv("ACLTRANSFORMER_HOME_PATH")) + "/tensors/thread_" + ss.str();
    size_t pos = filePath.find(basePath);
    std::string result = filePath;
    if (pos != std::string::npos) {
        result.erase(pos, basePath.length());
    }
    if (!copyTable.count(result)) {
        return;
    }

    ASD_LOG(INFO) << "save asdtensor start, tensor:" << AsdOpsTensorToString(tensor) << ", filePath:" << filePath;
    AsdOps::BinFile binFile;
    binFile.AddAttr("format", std::to_string(tensor.desc.format));
    binFile.AddAttr("dtype", std::to_string(tensor.desc.dtype));
    binFile.AddAttr("dims", AsdOpsDimsToString(tensor.desc.dims));
    if (tensor.data) {
        std::vector<char> hostData(tensor.dataSize);
        int st =
            AsdRtMemCopy(hostData.data(), tensor.dataSize, tensor.data, tensor.dataSize, ASDRT_MEMCOPY_DEVICE_TO_HOST);
        ASD_LOG_IF(st != 0, ERROR) << "AsdRtMemCopy device to host fail for save tensor, ret:" << st;
        binFile.AddObject("data", hostData.data(), tensor.dataSize);
    } else {
        ASD_LOG(INFO) << "save asdtensor " << filePath << " data is empty";
    }
    AsdOps::Status st = binFile.Write(filePath);
    if (st.Ok()) {
        ASD_LOG(INFO) << "save asdtensor " << filePath;
    } else {
        ASD_LOG(ERROR) << "save asdtensor " << filePath << " fail, error:" << st.Message();
    }
} 