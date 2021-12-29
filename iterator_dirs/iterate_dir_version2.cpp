#include <string>
#include <iostream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

int main() {
    std::string path = "pinghua_ceshi_images_dir_98_pts_dir_74";
    for (const auto & entry : fs::directory_iterator(path)){
      if (fs::is_regular_file(entry.status())  && !fs::is_symlink(entry.status())){
        std::string a_file_path(entry.path());
        std::cout<<a_file_path<<std::endl;
      }
    }
}

/*
there is some difference between experimental::filesystem and c++17 filesystem
execute: g++ -o test test.cpp --std=c++17 -lstdc++fs -Wall
refernce:
  1.https://stackoverflow.com/questions/60021128/c-experimental-filesystem-library-incomplete?rq=1
  2.https://stackoverflow.com/questions/67273/how-do-you-iterate-through-every-file-directory-recursively-in-standard-c
*/
