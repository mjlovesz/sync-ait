#ifndef ATB_PROBE_H
#define ATB_PROBE_H

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

namespace atb {
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
    static void SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath);
    static bool IsSaveTiling() const;
};
}
#endif