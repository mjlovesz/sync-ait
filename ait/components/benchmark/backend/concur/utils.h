#ifndef BACKEND_UTILS_H
#define BACKEND_UTILS_H

#include <string>
#include <vector>

using Arguments = std::unordered_map<std::string, std::string>;

void readArgs(int argc, char *argv[], Arguments& arguments);
std::stirng merge(std::vector<std::string> list, std::string delimiter); // merge a vector of string with delimiter
std::vector<size_t> strVecToNumVec(const std::vector<std::string>& vec);
std::vector<std::string> traversal(const char* dir);




#endif // BACKEND_UTILS_H