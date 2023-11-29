#include "atb_probe.h"
#incldue "binfile.h"
static std::vector<std::string> SplitString(const std::string &ss, const char &tar)
{
    std::vector<std::string> tokens;
    std::stringstream input(ss);
    std::string token;
    while (std::getline(input, token, tar))
    {
        tokens.push_back(token);
    }

    return tokens;

}


bool Probe::IsTensorNeedSave(const std::vector<int64_t> &ids, std::string &optype) const
{
    const char *vid = std::getenv("ATB_SAVE_TENSOR_IDS"); // 应该是20_1_9,1_23,5_29_1
    const char *tid = std::getenv("ATB_SAVE_TENSOR_RUNNER"); // 应该是LinearOps，SelfAttention
    if (!vid && !tid)
    {
        return true;
    }
    // 先用逗号分隔vid和tid
    std::vector<std::string> splitVid = SplitString(vid, ',');
    std::string splitTid = SplitString(tid, ',');
    std::string query = "";
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i)
        {
            query += "_" + to_string(ids[i]);
        }
        else 
        {
            query += to_string(ids[i]);
        }
    }
    for (auto &indice : splitVid)
    {
        if (indice == query) {
            return true;
        }
    }
    for (auto &indice : splitTid)
    {
        if (indice == optype) {
            return true;
        }
    }
    return false;
}


bool Probe::IsSaveTensorData() const
{
    const char* saveTensor = std::getenv("ATB_SAVE_TENSOR");
    if (saveTensor == "1")
    {
        return true;
    }
    return false;
}


bool Probe::IsSaveTensorDesc() const
{
    return true;
}


bool Probe::IsExecuteCountInRange(const uint64_t executeCount) const
{
    const char* saveTensorRange = std::getenv("ATB_SAVE_TENSOR_RANGE");
    std::vector<std::string> saveTensorRan = SplitString(saveTensorRange, ',');
    for (size_t i = 1; i < saveTensorRan.size(); i += 2) {
        uint64_t left = stoi(saveTensorRan[i - 1]), right = stoi(saveTensorRan[i]);
        if (executeCount <= right && executeCount >= left) {
            return true;
        }
    }
    return false;
}


bool Probe::IsSaveTensorBefore() const
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    if (saveTensorTime == "0" || saveTensorTime == "2") {
        return true;
    }
    return false;
}


bool Probe::IsSaveTensorAfter() const
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    if (saveTensorTime == "1" || saveTensorTime == "2") {
        return true;
    }
    return false;
}


void Probe::SaveTensor(const std::string &format, const std::string &dtype,
        const std::string &dims, const void *hostData, uint64_t dataSize,
        const std::string &filePath)
{   int flag = std::stoi(std::getenv("LOG_TO_STDOUT"));
    if (!hostData)
    {   
        if (flag) {
            std::cout << "hostData is None." << std::endl;
        }
        return;
    }
    BinFile binFile;
    binFile.AddAttr("format", format);
    binFile.AddAttr("dtype", dtype);
    binFile.AddAttr("dims", dims);
    binFile.AddObject("data", hostData, dataSize);
    binFile.Write(filePath);

}


void Probe::SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath)
{
    std::ofstream outfile(filePath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile.write(data, dataSize);
        outfile.close();
        std::cout << "Data written to file successfully!" << std::endl;
    } else {
        std::cout << "Unable to open file!" << std::endl;
    }
}


bool Probe::IsSaveTiling() const
{
    const char* isSaveTiling = std::getenv("ATB_SAVE_TILING");
    if (isSaveTiling == "1") {
        return true;
    }
    return false;
}