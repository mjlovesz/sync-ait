#ifndef ATB_PROBE_H
#define ATB_PROBE_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

namespace atb {
class Probe{
public:
    static bool IsTensorNeedSave(const std::vector<int64_t> &ids, const std::string &optype);
    static bool IsSaveTensorData();
    static bool IsSaveTensorDesc();
    static bool IsExecuteCountInRange(const uint64_t executeCount);
    static bool IsSaveTensorBefore();
    static bool IsSaveTensorAfter();
    static void SaveTensor(const std::string &format, const std::string &dtype,
        const std::string &dims, const void *deviceData, uint64_t dataSize,
        const std::string &filePath);
    static void SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath);
    static bool IsSaveTiling();
};
}
#endif