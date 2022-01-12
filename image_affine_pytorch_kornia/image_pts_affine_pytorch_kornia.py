import os
import sys
import glob
import shutil
import numpy as np
import cv2
import math
import torch
import kornia.geometry.transform as kgt

def read_pts(a_pts_path):
    points = []
    for a_line  in open(a_pts_path):
        a_line = a_line.strip()
        array = a_line.split()
        array = [float(x) for x in array]       
        points.append(array) 
    points = np.asarray(points)
    return points

if __name__=="__main__":
    image_name = "7_Cheering_Cheering_7_81.jpg"
    pts_name = "7_Cheering_Cheering_7_81.pts"

    points = read_pts(pts_name)
    a_image = cv2.imread(image_name)
    h, w, _ = a_image.shape
      
    angle = 20
      
    #------cv2------
    M = cv2.getRotationMatrix2D((w/2., h/2.), angle*1., 1.0) #--以图像中心进行旋转
    #print("cv2:M")
    #print(M)
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = h * abs(sin) + w * abs(cos)
    b_h = h * abs(cos) + w * abs(sin)
    M[0, 2] += ((b_w / 2.) - w/2.)
    M[1, 2] += ((b_h / 2.) - h/2.)
    print(M)
    a_image_cv = a_image.copy()
    a_image_cv = cv2.warpAffine(a_image_cv, M, (int(b_w), int(b_h)))
    print("a_image_cv.shape:\t", a_image_cv.shape)

    #----旋转关键点----
    #1.先去掉平移
    offset = np.array([w/2., h/2.])
    points_cv = points - offset
    #2.进行旋转
    M = M[:,:2].T
    #print(M)
    points_cv = np.dot(points_cv, M)
    #3.加上平移
    points_cv = points_cv + np.asarray([b_w/2., b_h/2.])

    for i in range(98):
        x = points_cv[i][0]
        y = points_cv[i][1]
        x = int(x); y = int(y)
        color = (255, 255, 255)
        cv2.circle(a_image_cv, (x, y), 1, color, -1)
    cv2.imwrite("hh.jpg", a_image_cv)
    #------cv2------

    #----pytorch-----
    a_image_k = torch.tensor(a_image).permute(2,0,1)
    a_image_k = a_image_k.unsqueeze(0)
    center = torch.tensor([[w/2, h/2]])
    angle = torch.tensor([angle*1.])
    scale = torch.tensor([[1., 1.]])
    M = kgt.get_rotation_matrix2d(center, angle, scale) 
    #print("kornia.M")
    #print(M)
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = h * abs(sin) + w * abs(cos)
    b_h = h * abs(cos) + w * abs(sin)
    M[0][0][2] += ((b_w / 2.) - w/2.)
    M[0][1][2] += ((b_h / 2.) - h/2.) 
    #print(M)

    a_image_k = a_image_k.float()
    a_image_k = kgt.warp_affine(a_image_k, M, (int(b_w), int(b_h))) 
    a_image_k = a_image_k.squeeze().permute(1,2,0)
    a_image_k = a_image_k.numpy().astype(np.uint8)
    a_image_k = a_image_k.copy()

    #1.先去掉平移
    offset = np.array([w/2., h/2.])
    points_k = points - offset
    #2.进行旋转
    M = M[:,:,:2].squeeze().t().numpy()
    print(M)
    points_k = np.dot(points_k, M)
    #3.加上平移
    points_k = points_k + np.asarray([b_w/2., b_h/2.])

    for i in range(98):
        x = points_k[i][0]
        y = points_k[i][1]
        x = int(x); y = int(y)
        color = (0, 255, 0)
        cv2.circle(a_image_k, (x, y), 1, color, -1)
    cv2.imwrite("hh_k.jpg", a_image_k)
