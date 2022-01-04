#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
using namespace std;

int main() {
  std::string a_file_name = "aaa/bbbb/cccc/dddd.jpg";
  //regex, c++11
  std::regex re("/");
  std::vector<std::string> tokens(
      std::sregex_token_iterator(a_file_name.begin(), a_file_name.end(), re, -1),
      std::sregex_token_iterator()
  );
  for(auto s:tokens) std::cout<<s<<std::endl;
}
