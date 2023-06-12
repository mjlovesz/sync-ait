#ifndef BACKEND_UTILS_H
#define BACKEND_UTILS_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>

namespace chr = std::chrono;
using TimePointPair = std::pair<chr::steady_clock::time_point, chr::steady_clock::time_point>;
using Arguments = std::unordered_map<std::string, std::string>;

void readArgs(int argc, char *argv[], Arguments& arguments);
std::string merge_str(std::vector<std::string> list, std::string delimiter); // merge a vector of string with delimiter
std::vector<std::string> split_str(std::string input, char delimiter);
std::vector<size_t> strVecToNumVec(const std::vector<std::string>& vec);
std::vector<std::string> traversal(const char* dir); // traversal a directory return vector of filename
int createFilesList(std::vector<std::vector<std::string>>& fileList, std::string input);
std::string getPrefix(std::string filePath);
std::string removeSlash(std::string name);
std::string createDynamicShape(std::string name, std::vector<size_t> shapes);
void printTimeWall(const std::string& phase, const std::vector<TimePointPair>& timestamps);





#endif // BACKEND_UTILS_H