#include <iostream>
#include <string>
#include <numeric>


#include "utils.h"


void readArgs(int argc, char *argv[], Arguments& arguments)
{
    for (int i = 1; i < argc; ++i) {
        auto valuePtr = strchr(argv[i], '=');
        if (valuePtr) {
            std::string value{valuePtr + 1};
            std::string key{argv[i], valuePtr - argv[i]};
            if (arguments.find(key) != arguments.end()) {
                arguments[key] = value;
            } else {
                std::cout << key << " is not a parameter" << std::endl;
            }
        } else {
            std::cout << "pass the parameter in form of key=value" << std::endl;
        }
    }
}


std::stirng merge(std::vector<std::string> list, std::string delimiter)
{
    auto res = std::accumulate(list.begin(), list.end(), std::string(),
    [=](const std::string& a, const std::string& b) -> std::string {
        return a + (a.length() > 0 ? delimiter : "") + b;
    } );
    return res;
}

std::vector<size_t> strVecToNumVec(const std::vector<std::string>& vec)
{
    std::vector<size_t> res;
    for (auto &elem: vec) {
        res.push(stoi(elem));
    }
    return res;
}