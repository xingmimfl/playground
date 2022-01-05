#include <string>
#include <iostream>
#include <regex>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include "OneEuroFilter.hpp"
#define PI 3.1415926
namespace fs = std::experimental::filesystem;
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

std::string getName(const string & a_file_name){
  std::regex re("/");
  std::vector<std::string> tokens(
      std::sregex_token_iterator(a_file_name.begin(), a_file_name.end(), re, -1),
      std::sregex_token_iterator()
  );
  return tokens[tokens.size()-1];
}


//---从关键点得到矩形框
cv::Rect get_rect_from_pts(const std::vector<cv::Point2f> & landmarks){
  int length = landmarks.size();
  float x1 = 10e5; float y1 = 10e5;
  float x2 = 10e-5; float y2 = 10e-5;
  for(int i=0; i<length; i++){
    cv::Point2f a_point = landmarks[i];
    float x = a_point.x; float y = a_point.y;
    x1 = x < x1 ? x : x1;
    x2 = x > x2 ? x : x2;
    y1 = y < y1 ? y : y1;
    y2 = y > y2 ? y : y2;
  }
  x1 = (int)x1; x2 = (int)x2;
  y1 = (int)y1; y2 = (int)y2;
  int width = x2 - x1 + 1;
  int height = y2 - y1 + 1;
  cv::Rect rect(x1, y1, width, height);
  return rect;
}

//---外扩rect区域
int expandBox(cv::Rect& rectBox, float expandratio){
  //不保证输出不越界,相对半径的比率
  float centerX = rectBox.x + 0.5 * rectBox.width;
  float centerY = rectBox.y + 0.5 * rectBox.height;

  //最大边
  float trg_size = std::max(rectBox.width, rectBox.height);
  float magic_number = 1.0;//目前阶段需要手工设定
  trg_size = trg_size * magic_number;

  //新半径。trg_size平均直径，短径*(1+扩展率)或长径*（1+0）
  //=等于各方向新的半径。
  float lexpand = 0.5f * trg_size * (1 + expandratio);
  float rexpand = 0.5f * trg_size * (1 + expandratio);
  float texpand = 0.5f * trg_size * (1 + expandratio);
  float bexpand = 0.5f * trg_size * (1 + expandratio);

  rectBox.x = centerX - lexpand;
  rectBox.y = centerY - texpand;
  rectBox.width = (1 + expandratio) * trg_size;
  rectBox.height = (1 + expandratio) * trg_size;
  return 0;
}

//得到landmark在rect里面的坐标
std::vector<cv::Point2f> getPtsInRect(const cv::Rect & rect, const std::vector<cv::Point2f> &landmarks){
  std::vector<cv::Point2f> ans;
  float x0 = rect.x;
  float y0 = rect.y;
  for(auto p: landmarks){
    cv::Point2f a_point;
    a_point.x = p.x - x0;
    a_point.y = p.y - y0;
    ans.emplace_back(a_point);
  } 
  return ans;
}

int main() {
  std::string path = "pinghua_ceshi_images_dir_98_pts_dir_74";
  std::string image_dir_path = "pinghua_ceshi_images_dir";
  //----read all pts files----
  std::vector<std::vector<cv::Point2f>> points_vec;
  std::vector<cv::Mat> images_vec;
  std::vector<std::string> image_names_vec;
  
  for (const auto & entry : fs::directory_iterator(path)){
    if (fs::is_regular_file(entry.status())  && !fs::is_symlink(entry.status())){
      std::string a_file_path(entry.path());
      vector<cv::Point2f> lmk;
      read_pts(a_file_path, lmk);
      points_vec.emplace_back(lmk);

      std::string a_pts_name = getName(a_file_path);
      std::string a_image_name = a_pts_name.substr(0, a_pts_name.size()-4) + ".jpg";
      image_names_vec.emplace_back(a_image_name);
      std::string a_image_path = image_dir_path + "/" + a_image_name;
      cv::Mat a_image = cv::imread(a_image_path);
      images_vec.emplace_back(a_image); 
    }
  }

  //---crop rect and resize to [0, 1]
  std::vector<cv::Rect> rect_vec;
  std::vector<cv::Mat> rect_image_vec;
  std::vector<std::vector<cv::Point2f>> rect_points_vec;
  for(int i=0; i<points_vec.size(); i++){
    std::vector<cv::Point2f> a_point = points_vec[i];
    cv::Mat a_image = images_vec[i];
    //std::string a_image_name = image_names_vec[i];
    //for(auto p: a_point){
    //  cv::circle(a_image, cv::Point(int(p.x), int(p.y)), 1, cv::Scalar(255, 255, 255), -1);
    //}
    //cv::imwrite(a_image_name, a_image);
    cv::Rect rect = get_rect_from_pts(a_point); 
    expandBox(rect, 0.3);
    std::vector<cv::Point2f> a_rect_point = getPtsInRect(rect, a_point);
    cv::Mat a_rect_image = images_vec[i](rect);
    
    rect_vec.emplace_back(rect);
    rect_image_vec.emplace_back(a_rect_image); 
    rect_points_vec.emplace_back(a_rect_point);
    //for(auto p: a_rect_point){
    //  cv::circle(a_rect_image, cv::Point(int(p.x), int(p.y)), 1, cv::Scalar(255, 255, 255), -1); 
    //}
    //std::string a_image_name = image_names_vec[i];
    //cv::imwrite(a_image_name, a_rect_image);   
  }    
  
  //----one euro filter----
  //---先使用相同的值----
  float mincutoff = 0.008;
  float beta = 100;
  float frequency = 20;
  float dcutoff = 1.0;
  std::vector<OneEuroFilter *> filters_vec;
  filters_vec.reserve(148);
  for(int i=0; i<148; i++){
    filters_vec.emplace_back(new OneEuroFilter(frequency, mincutoff, beta, dcutoff));
  }

  //-----开始滤波----
  float coef = 0.1;
  //float Scale = 128.;
  std::vector<std::vector<cv::Point2f>> new_rect_points_vec;
  for(int i=0; i<rect_points_vec.size(); i++){
    std::vector<cv::Point2f> a_point = rect_points_vec[i];
    cv::Mat a_image = rect_image_vec[i];
    float face_size = (a_image.rows + a_image.cols) / 2.;
    float Scale = face_size;
    for(int j=0; j<148; j++){
      float x0 = a_point[j].x / Scale;
      float y0 = a_point[j].y / Scale;
      float x1 = filters_vec[j]->filter(x0);
      float y1 = filters_vec[j]->filter(y0);
      a_point[j].x = x1 * Scale * coef + a_point[j].x * (1.0 - coef);
      a_point[j].y = y1 * Scale * coef + a_point[j].y * (1.0 - coef);
    }
    new_rect_points_vec.emplace_back(a_point);
  }  

  for(int i=0; i<new_rect_points_vec.size(); i++){
    std::string a_image_name = image_names_vec[i];
    std::vector<cv::Point2f> a_point = new_rect_points_vec[i];
    cv::Mat a_image = images_vec[i];
    cv::Rect a_rect = rect_vec[i];
    for(auto p: a_point){
      p.x += a_rect.x;
      p.y += a_rect.y;
      cv::circle(a_image, cv::Point(int(p.x), int(p.y)), 1, cv::Scalar(255, 255, 255), -1);
    }
    cv::imwrite(a_image_name, a_image);
  } 
  return 0;
}
