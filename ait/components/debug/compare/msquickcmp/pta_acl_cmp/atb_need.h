#ifndef ATB_NEED_H
#define ATB_NEED_H
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include <cstdint>

#include <openssl/md5.h>
#include <openssl/evp.h>
#include <syscall.h>
#include <sys/stat.h>


namespace AsdOps {
constexpr size_t MAX_SVECTOR_SIZE = 48;
constexpr bool CHECK_BOUND = true;

template <class T, std::size_t MAX_SIZE = MAX_SVECTOR_SIZE> class SVector {
private:
    T storage_[MAX_SIZE + 1];
    std::size_t size_{0};
};


enum TensorDType : int {
    TENSOR_DTYPE_UNDEFINED = -1,
    TENSOR_DTYPE_FLOAT = 0,
    TENSOR_DTYPE_FLOAT16 = 1,
    TENSOR_DTYPE_INT8 = 2,
    TENSOR_DTYPE_INT32 = 3,
    TENSOR_DTYPE_UINT8 = 4,
    TENSOR_DTYPE_INT16 = 6,
    TENSOR_DTYPE_UINT16 = 7,
    TENSOR_DTYPE_UINT32 = 8,
    TENSOR_DTYPE_INT64 = 9,
    TENSOR_DTYPE_UINT64 = 10,
    TENSOR_DTYPE_DOUBLE = 11,
    TENSOR_DTYPE_BOOL = 12,
    TENSOR_DTYPE_STRING = 13,
    TENSOR_DTYPE_COMPLEX64 = 16,
    TENSOR_DTYPE_COMPLEX128 = 17,
    TENSOR_DTYPE_BF16 = 27
};


enum TensorFormat : int {
    TENSOR_FORMAT_UNDEFINED = -1,
    TENSOR_FORMAT_NCHW = 0,
    TENSOR_FORMAT_NHWC = 1,
    TENSOR_FORMAT_ND = 2,
    TENSOR_FORMAT_NC1HWC0 = 3,
    TENSOR_FORMAT_FRACTAL_Z = 4,
    TENSOR_FORMAT_NC1HWC0_C04 = 12,
    TENSOR_FORMAT_HWCN = 16,
    TENSOR_FORMAT_NDHWC = 27,
    TENSOR_FORMAT_FRACTAL_NZ = 29,
    TENSOR_FORMAT_NCDHW = 30,
    TENSOR_FORMAT_NDC1HWC0 = 32,
    TENSOR_FORMAT_FRACTAL_Z_3D = 33
};


struct TensorDesc {
    TensorDType dtype = TENSOR_DTYPE_UNDEFINED;
    TensorFormat format = TENSOR_FORMAT_UNDEFINED;
    AsdOps::SVector<int64_t> dims;
};


struct Tensor {
    TensorDesc desc;
    void *data = nullptr;
    size_t dataSize = 0;
    size_t pos = 0;
    void *hostData = nullptr;
};


class BinFile {
public:
    BinFile();
    ~BinFile();
    char* AddAttr(const std::string &name, const std::string &value);
    char* Write(const std::string &filePath);
private:
    struct Object {
        uint64_t offset = 0;
        uint64_t length = 0;
    };
};
}


namespace atb {
    constexpr uint32_t MAX_DIM = 8;
    struct Dims {
        int64_t dims[MAX_DIM];
        uint64_t dimNum = 0;
    };
    struct TensorDesc {
        int dtype = -1;
        int format = -1;
        Dims shape;
    };
    struct Tensor {
        TensorDesc desc;
        void *deviceData = nullptr;
        void *hostData = nullptr;
        uint64_t dataSize = 0;
    };
    class TensorUtil {
    public:
        static std::string ShapeToString(const Dims &dims);
        static std::string AsdOpsDimsToString(const AsdOps::SVector<int64_t> &dims);
    };
    class StoreUtil {
    private:
        static void SaveTensor(const AsdOps::Tensor &tensor, const std::string &filePath);
        static void SaveTensor(const Tensor &tensor, const std::string &filePath);
        static void SaveTensor(const std::string &format, const std::string &dtype, const std::string &dims,
            const void *deviceData, uint64_t dataSize, const std::string &filePath);
    };
}
#endif