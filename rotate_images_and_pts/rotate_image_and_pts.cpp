#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#define pi 3.1415926
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



float getAnglefromPts(const cv::Point2f &ptA, const cv::Point2f &ptB){
  float angle = std::atan2(ptA.y-ptB.y, ptA.x-ptB.x);    
  angle = angle / pi * 180.;
  angle = 180 - angle;
  angle = -angle;
  return angle;
}

int main(){
  const string a_image_path = "qingxie_angle_image_pts/xxxx.jpg";
  const string a_pts_path = "qingxie_angle_image_pts/xxxx.pts";
  cv::Mat a_image = cv::imread(a_image_path);
  std::vector<cv::Point2f> landmarks;
  read_pts(a_pts_path, landmarks);
  cv::Rect rect = get_rect_from_pts(landmarks); 
  expandBox(rect, 0.3);

  //--crop_image, transform landmarks
  cv::Mat a_crop_image = a_image(rect); 
  for(int i = 0; i<landmarks.size(); i++){
    landmarks[i].x -= rect.x;  
    landmarks[i].y -= rect.y;
  }
  int crop_h = a_crop_image.rows;
  int crop_w = a_crop_image.cols;
  //get angle
  cv::Point2f ptA = landmarks[45];//左内眼角
  cv::Point2f ptB = landmarks[51];//右外眼角
  float angle = getAnglefromPts(ptA, ptB);

  //we have to use ((crop_w-1)/2, (crop_h-1)/2)， I dont know why
  //many examples of c++ write in this way
  cv::Mat M = cv::getRotationMatrix2D(cv::Point2f((crop_w-1)/2., (crop_h-1)/2.), angle, 1.0); //--以图像中心进行旋转
  cv::warpAffine(a_crop_image, a_crop_image, M, cv::Size(crop_w, crop_h));

  //remove offset 
  for(int i=0; i<landmarks.size(); i++){
    landmarks[i].x -= crop_w/2;
    landmarks[i].y -= crop_h/2;      
  }

  for(int i=0; i<landmarks.size(); i++){
      cv::Point2f a_point = landmarks[i];
      float x = a_point.x;
      float y = a_point.y;
      //caution!!!! must use double type here, float is not right!!!!
      landmarks[i].x = x * M.at<double>(0,0) + y * M.at<double>(0,1);
      landmarks[i].y = x * M.at<double>(1,0) + y * M.at<double>(1,1);
  }
  //add offset
  for(int i=0; i<landmarks.size(); i++){
    landmarks[i].x += crop_w/2;
    landmarks[i].y += crop_h/2;
  }

  //draw pts
  for(int i = 0; i<landmarks.size(); i++){
    cv::Point2f a_point = landmarks[i];
    cv::circle(a_crop_image, cv::Point(int(a_point.x), int(a_point.y)), 1, cv::Scalar(255, 255, 255), -1); 
  } 
  //draw rect
  //cv::rectangle(a_crop_image, rect, cv::Scalar(0, 255,255), 1); 
  cv::imwrite("hh.png", a_crop_image);
}
