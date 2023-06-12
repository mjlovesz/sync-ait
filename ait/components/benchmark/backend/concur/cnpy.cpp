#include "cnpy.h"
#include <typeinfo>
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <regex>


char cnpy::BigEndianTest()
{
    int x = 1;
    return (((char *)&x)[0]) ? '<' : '>';
}

char cnpy::map_type(const std::type_info &t)
{
    if (t == typeid(float) || t == typeid(double) || t == typeid(long double))
        return 'f';
        
    if (t == typeid(int) || t == typeid(char) || t == typeid(short) || t == typeid(long) || t == typeid(long long))
        return 'i';
    if (t == typeid(unsigned char) || t == typeid(unsigned short) || t == typeid(unsigned long) ||
        t == typeid(unsigned long long) || t == typeid(unsigned int))
        return 'u';
    
    if (t == typeid(bool))
        return 'b';
    
    if (t == typeid(std::complex<float>) || t == typeid(std::complex<double>) || t == typeid(std::complex<long double>))
        return 'c';
    else
        return '?';
}

template <> std::vector<char> &cnpy::operator += (std::vector<char> &lhs, const std::string rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template <> std::vector<char> &cnpy::operator += (std::vector<char> &lhs, const char *rhs)
{
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

void cnpy::parse_npy_header(unsigned char *buffer, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order)
{
    uint16_t header_len = *reinterpret_cast<uint16_t *>(buffer + 8);            //8 means offset of header_len
    std::string header(reinterpret_cast<char *>(buffer + 9), header_len);       // 9 means offser of header

    size_t loc1, loc2;

    loc1 = header.find("fortran_order") + 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();
    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }
    
    loc1 = header.find("descr") + 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
    
}

void cnpy::parse_npy_header(FILE *fp, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order)
{
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    size_t loc1, loc2;

    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword : 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true :false);

    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword : 'descr'");
    loc1 += 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
}

cnpy::NpyArray load_the_npy_file(FILE *fp)
{
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(fp, word_size, shape, fortran_order);

    cnpy::NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

cnpy::NpyArray cnpy::npy_load(std::string fname)
{
    FILE *fp = fopen(fname.c_str(), "rb");

    if (!fp)
        throw std::runtime_error("npy_load: Unable to open file" + fname);
    
    NpyArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}

