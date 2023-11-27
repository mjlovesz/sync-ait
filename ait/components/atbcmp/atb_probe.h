#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

class Probe{
public:
    static bool IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &name) const;
    static bool IsSaveTensorData() const;
    static bool IsSaveTensorDesc() const;
    static bool IsExecuteCountInRange(const uint64_t executeCount) const;
    static bool IsSaveTensorBefore() const;
    static bool IsSaveTensorAfter() const;
    static void SaveTensor(const std::string &format, const std::string &dtype,
        const std::string &dims, const void *deviceData, uint64_t dataSize,
        const std::string &filePath);
    static bool IsSaveTiling() const;
    static void SaveTiling(uint8_t *hostData, uint64_t dataSize, const std::string &filePath);
}