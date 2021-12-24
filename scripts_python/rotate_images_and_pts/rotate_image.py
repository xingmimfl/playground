import os
import sys
import shutil
import cv2
import numpy as np
import math

def read_pts(a_pts_path):
  points = []
  for a_line in open(a_pts_path):
    a_line = a_line.strip()
    array = a_line.split()
    array = [float(x) for x in array]
    points.append(array)
  points = np.asarray(points)
  return points

if __name__=='__main__':
    a_image_path = "qingxie_angle_image_pts/xxx.jpg" 
    a_pts_path = "qingxie_angle_image_pts/xxx.pts"
    a_image = cv2.imread(a_image_path)
    h, w, _ = a_image.shape

    points = read_pts(a_pts_path)
    #using inner point of left eye and inner point of right eye
    #other points would also be ok
    left_waiyanjiao = points[45]
    right_waiyanjiao = points[51]
    cv2.circle(a_image, (int(left_waiyanjiao[0]), int(left_waiyanjiao[1])), 2, (255, 255, 0), -1)
    cv2.circle(a_image, (int(right_waiyanjiao[0]), int(right_waiyanjiao[1])), 2, (255, 255, 0), -1)

    #----计算角度---
    angle = math.atan2(left_waiyanjiao[1]-right_waiyanjiao[1], left_waiyanjiao[0]-right_waiyanjiao[0])
    angle = angle / math.pi * 180
    angle = 180 - angle
    angle = -angle

    #one drawback of scale = 1 is that, after rotation, some border corner of original image would be discarded
    scale = 1.0.
    #---旋转矩阵/图片----
    new_w = int(scale * w)
    new_h = int(scale * h)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0) #--以图像中心进行旋转
    a_image = cv2.warpAffine(a_image, M, (new_w, new_h))

    #----旋转关键点----
    #1.先去掉平移
    offset = np.array([w/2, h/2])
    points = points - offset
    #2.进行旋转
    M = M[:,:2].T
    points = np.dot(points, M)
    #3.加上平移
    points = points + offset

    for i in range(101):
        x = points[i][0]
        y = points[i][1]
        x = int(x); y = int(y)
        color = (255, 255, 255)
        cv2.circle(a_image, (x, y), 1, color, -1)
    cv2.imwrite("hh.jpg", a_image)
