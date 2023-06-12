#include <iostream>
#include <sstream>
#include <string>
#include <numeric>
#include <dirent.h>

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

std::vector<std::string> split(std::string input, char delimiter)
{
    std::stringstream ss(input);
    std::vector<std::string> res;

    while (ss.good()) {
        std::string substr;
        getline(ss, substr, delimiter);
        res.push_back(substr);
    }
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

std::vector<std::string> traversal(const char* dir)
{
    DIR *dir_ptr;
    struct dirent *diread;
    if (dir_ptr = opendir(dir) != nullptr) {
        while (diread = readdir(dir_ptr) != nullptr) {
            if (diread->d_type == DT_REG)
                filenames.push_back(string(dir) + diread->d_name);
        }
        closedir(dir_ptr);
    }
    sort(filenames.begin(), filenames.end());
    return filenames;
}

// input need to be dir1,dir2,dir3,...
int createFilesList(std::vector<std::vecotr<std::string>>& fileList, std::string input)
{
    std::vector<std::vector<std::string>> directorys;
    
    for (auto &dir: split(input, ',')) {
        if (dir.back() != '/') {
            dir.push_back('/');
        }
        directorys.push_back(std::move(traversal(dir.c_str())));
    }
    // check whether number of files in each directory is the same
    size_n = directorys[0].size();
    for (auto &directory: directorys) {
        if (directory.size() != n) {
            return 1;
        }
    }

    for (size_t i = 0; i < n; i++) {
        std::vector<std::string>> combine;
        for (auto &directory : directorys) {
            combine.push_back(directory[i]);
        }
        fileList.push_back(std::move(combine));
    }
    return 0;

}

std::string getPrefix(std::string filePath)
{
    std::stringstream ss(filePath);
    std::string res{};
    while (ss.good()) {
        std::string substr;
        getline(ss, substr, '/');
        if (substr == "") {
            continue;
        }
        res = substr;
    }
    return res;
}

std::string removeSlash(std::string name)
{
    std::string res;
    for (auto &elem: name) {
        if (elem != '/') {
            res.push_back(elem);
        }
    }
    return res;
}

std::string createDynamicShape(std::string name, std::vector<size_t> shapes)
{
    std::vector<std::string> shapes_str{};
    for (auto &shape: shapes) {
        shapes_str.push_back(std::to_string(shape));
    }
    auto res = merge(shapes_str, ",");
    return name + ":" + res;
}

void printTimeWall(const std::string& phase, const std::vector<TimePointPair>& timestamps)
{
    if (timestamps.empty()) {
        return;
    }
    auto [min_it, max_it] = std::minmax_element(timestamps.begin(), timestamps.end(),
    [](auto tp1, auto tp2) {return tp1.second - tp1.first < tp2.second - tp2.first;});

    auto total std::accumulate(timestamps.begin(), timestamps.end(), 0,
    [](auto init, auto tpp) {return init + chr::duration_cast<chr::microseconds>(tpp.second - tpp.first).count(); });

    auto avg = total/timestamps.size();
    std::cout
        << phase << " avg: " << avg/1000.0 << "ms. min: "
        << chr::duration_cast<chr::microseconds>(min_it->second - min_it->first).count()/1000
        << "ms. max: " << chr::duration_cast<chr::microseconds>(max_it->second - max_it->first).count()/1000
        << "ms." << std::endl;
}