#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

void split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimiters = " ")
{
  std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
  std::string::size_type pos = s.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delimiters, pos);
    pos = s.find_first_of(delimiters, lastPos);
  }
}

int main() {
  std::string a_file_name = "aaa/bbbb/cccc/dddd.jpg";
  std::vector<std::string> tokens;
  split(a_file_name, tokens, "/");
  for(auto s: tokens)
    std::cout<<s<<std::endl;  
}
