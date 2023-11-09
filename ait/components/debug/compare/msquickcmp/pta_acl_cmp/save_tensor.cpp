#include "atb_need.h"
#include <acl/acl_rt.h>

std::string BufMd5(const unsigned char *buf, size_t buf_size)
{
    unsigned char hash[MD5_DIGEST_LENGTH];  // MD5_DIGEST_LENGTH is 16

    EVP_MD_CTX *mdctx;
    unsigned int md5_digest_len = EVP_MD_size(EVP_md5());

    // MD5_Init
    mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_md5(), nullptr);

    // MD5_Update
    EVP_DigestUpdate(mdctx, buf, buf_size);

    // MD5_Final
    std::array<uint8_t, MD5_DIGEST_LENGTH> result;
    EVP_DigestFinal_ex(mdctx, result.data(), nullptr);
    EVP_MD_CTX_free(mdctx);

    std::stringstream ss;
    for(int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(result[i]);
    }
    return ss.str();
}


void InitialPathTable(std::unordered_set<std::string> &pathTable)
{
    const char* envValue = std::getenv("AIT_CMP_TASK_PID");
    std::string taskPid = envValue ? envValue : "";
    if (taskPid != "") {
        taskPid = "/" + taskPid;
    }
    std::string fileName = "/tmp" + taskPid + "/ait_compare_acl_map.txt";
    std::ifstream fileContent(fileName);
    if (fileContent.is_open()) {
        std::string filePath;
        while (std::getline(fileContent, filePath)) {
            pathTable.insert(filePath);
        }
        fileContent.close();
    }
}


std::unordered_set<std::string> &FindTable()
{
    static std::unordered_set<std::string> pathTable;
    static bool initialized = false;
    if (!initialized) {
        InitialPathTable(pathTable);
        initialized = true;
    }
    return pathTable;
}

bool isPathInTable(const std::string &filePath)
{
    pid_t processID = getpid();
    std::string pID = std::to_string(processID);
    pID = pID + "_" + pID;
    const char* fileDir = std::getenv("ASDOPS_LOG_TO_FILE_DIR");
    std::string dataDir = fileDir ? fileDir : "";
    const char* aitTaskIdEnv = std::getenv("AIT_CMP_TASK_ID");
    std::string aitTaskId = aitTaskIdEnv ? std::string(aitTaskIdEnv) : "";
    std::string basePath = dataDir + "/tensors/" + pID + "/" + aitTaskId + "/";

    size_t pos = filePath.find(basePath);
    std::string result = filePath;
    if (pos != std::string::npos) {
        result.erase(pos, basePath.length());
    }

    std::string baseDir = dataDir + "/tensors/thread_";
    size_t originPos = filePath.find(baseDir);
    std::string originResult = filePath;
    if (originPos != std::string::npos) {
        size_t slashPos = filePath.find("/", originPos + basePath.length());
        originResult = filePath.substr(slashPos + 1);
    }

    std::unordered_set<std::string> &copyTable = FindTable();
    if (!copyTable.count(result) && !copyTable.count(originResult)) {
        return false;
    } else {
        return true;
    }
}

void SaveMd5ToFile(const void* deviceData, uint64_t dataSize, const std::string &filePath)
{
    if (!deviceData) {
        return;
    }

    std::vector<char> hostData(dataSize);
    int st =
        aclrtMemcpy(hostData.data(), dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::string md5 = BufMd5((unsigned char*)hostData.data(), dataSize);

    size_t sep_pos = filePath.rfind("/");
    std::string md5_filePath = filePath;
    if (sep_pos != std::string::npos) {
        md5_filePath.erase(sep_pos);
        md5_filePath = md5_filePath + "/" + md5;
    } else {
        md5_filePath = md5;
    }
    std::ofstream outfile(md5_filePath.c_str());
}


bool isInTensorBinPath(const std::string &filePath)
{
    size_t sep_pos = filePath.rfind("/");
    std::string fileName = filePath;
    if (sep_pos != std::string::npos) {
        fileName.erase(0, sep_pos + 1);
    }
    return fileName.find("intensor") != std::string::npos || fileName.find("inTensor") != std::string::npos;
}


void atb::StoreUtil::SaveTensor(const AsdOps::Tensor &tensor, const std::string &filePath)
{
    bool is_save_md5 = false;
    const char *envStr = std::getenv("AIT_IS_SAVE_MD5");
    if (envStr != nullptr && std::string(envStr) == "1") {
        is_save_md5 = true;
    }

    if (!is_save_md5) {
        if (!isPathInTable(filePath) ) {
            return;
        }
        SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
            TensorUtil::AsdOpsDimsToString(tensor.desc.dims), tensor.data, tensor.dataSize, filePath);
    } else if (isInTensorBinPath(filePath)) {
        SaveMd5ToFile(tensor.data, tensor.dataSize, filePath);
    } else {
        SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
            TensorUtil::AsdOpsDimsToString(tensor.desc.dims), tensor.data, tensor.dataSize, filePath);
    }
}


void atb::StoreUtil::SaveTensor(const Tensor &tensor, const std::string &filePath)
{
    bool is_save_md5 = false;
    const char *envStr = std::getenv("AIT_IS_SAVE_MD5");
    if (envStr != nullptr && std::string(envStr) == "1") {
        is_save_md5 = true;
    }

    if (!is_save_md5) {
        if (!isPathInTable(filePath) ) {
            return;
        }
        SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
            TensorUtil::ShapeToString(tensor.desc.shape), tensor.deviceData, tensor.dataSize, filePath);
    } else if (isInTensorBinPath(filePath)) {
        saveMd5ToFile(tensor.deviceData, tensor.dataSize, filePath);
    } else {
        SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
            TensorUtil::ShapeToString(tensor.desc.shape), tensor.deviceData, tensor.dataSize, filePath);
    }
}
