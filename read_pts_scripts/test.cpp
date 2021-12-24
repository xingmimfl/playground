#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;

int read_pts(const string & a_pts_path, std::vector<cv::Point2f> & landmarks){
  ifstream in(a_pts_path);
  string a_line;
  while(getline(in, a_line)){
    stringstream ss(a_line);
    string inner_tmp;
    cv::Point2f a_point;
    int count = 0;
    while(getline(ss, inner_tmp, ' ')){//分割ss
      if(count==0){
        a_point.x = std::atof(inner_tmp.c_str());//string to float
        count++;
      }else{
        a_point.y = std::atof(inner_tmp.c_str());
        count++;
      }
    }
    landmarks.emplace_back(a_point);
  } 
  return 0;
}

int main(){
  const string a_image_path = "qingxie_angle_image_pts/xx.jpg";
  const string a_pts_path = "qingxie_angle_image_pts/xx.pts";
  cv::Mat a_image = cv::imread(a_image_path);
  std::vector<cv::Point2f> landmarks;
  read_pts(a_pts_path, landmarks);
  
  for(int i = 0; i<landmarks.size(); i++){
    cv::Point2f a_point = landmarks[i];
    cv::circle(a_image, cv::Point(int(a_point.x), int(a_point.y)), 3, cv::Scalar(0, 255, 0), -1); 
  } 
  cv::imwrite("hh.png", a_image);
}
