#include "atb_need.h"

std::string bufMd5(const unsigned char *buf, size_t buf_size)
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


std::unordered_set<std::string> &findTable()
{
    static std::unordered_set<std::string> pathTable;
    if (pathTable.empty()) {
        InitialPathTable(pathTable);
    }
    return pathTable;
}

bool isPathInTable(const std::string &filePath)
{
    pid_t processID = getpid();
    std::string pID = std::to_string(processID);
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

    std::unordered_set<std::string> &copyTable = findTable();
    if (!copyTable.count(result) && !copyTable.count(originResult)) {
        return false;
    } else {
        return true;
    }
}

void saveMd5ToFile(const AsdOps::Tensor &tensor, const std::string &filePath, const std::string tensorDimsStr)
{
    if (!tensor.data) {
        ATB_LOG(INFO) << "save asdtensor " << filePath << " data is empty";
        return;
    }

    AsdOps::BinFile binFile;
    binFile.AddAttr("format", std::to_string(tensor.desc.format));
    binFile.AddAttr("dtype", std::to_string(tensor.desc.dtype));
    binFile.AddAttr("dims", tensorDimsStr);

    std::vector<char> hostData(tensor.dataSize);
    int st =
        AsdRtMemCopy(hostData.data(), tensor.dataSize, tensor.data, tensor.dataSize, ASDRT_MEMCOPY_DEVICE_TO_HOST);
    ATB_LOG_IF(st != 0, ERROR) << "AsdRtMemCopy device to host fail for save tensor, ret:" << st;

    std::string md5 = bufMd5((unsigned char*)hostData.data(), tensor.dataSize);

    size_t sep_pos = filePath.rfind("/");
    std::string md5_filePath = filePath;
    if (sep_pos != std::string::npos) {
        md5_filePath.erase(sep_pos);
        md5_filePath = md5_filePath + "/" + md5;
    } else {
        md5_filePath = md5;
    }
    ATB_LOG(INFO) << "write md5 info for intensor, md5: " << md5;

    AsdOps::Status write_st = binFile.Write(md5_filePath);
    if (write_st.Ok()) {
        ATB_LOG(INFO) << "save md5 " << md5_filePath;
    } else {
        ATB_LOG(ERROR) << "save md5 " << md5_filePath << " fail, error:" << write_st.Message();
    }
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


void AclTransformer::StoreUtil::SaveTensor(const AsdOps::Tensor &tensor, const std::string &filePath)
{
    bool is_save_md5 = false;
    const char *envStr = std::getenv("AIT_IS_SAVE_MD5");
    if (envStr != nullptr && std::string(envStr) == "1") {
        is_save_md5 = true;
    }

    if (!is_save_md5) {
        if (! isPathInTable(filePath) ) {
            return;
        }
        SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
            TensorUtil::AsdOpsDimsToString(tensor.desc.dims), tensor.data, tensor.dataSize, filePath);
    } else if (isInTensorBinPath(filePath)) {
        saveMd5ToFile(tensor, filePath, AsdOpsDimsToString(tensor.desc.dims));
    } else {
        SaveTensor(std::to_string(tensor.desc.format), std::to_string(tensor.desc.dtype),
            TensorUtil::AsdOpsDimsToString(tensor.desc.dims), tensor.data, tensor.dataSize, filePath);
    }
}