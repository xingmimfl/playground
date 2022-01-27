import os
import sys
import glob
import shutil
import numpy as np
import cv2
import math
import torch
#----http://nghiaho.com/?page_id=671

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    #if np.linalg.det(R) < 0:
    #    print("det(R) < R, reflection detected!, correcting for it ...")
    #    Vt[2,:] *= -1
    #    R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    return R, t

def read_pts(a_pts_path):
    points = []
    for a_line  in open(a_pts_path):
        a_line = a_line.strip()
        array = a_line.split()
        array = [float(x) for x in array]       
        points.append(array) 
    points = np.asarray(points)
    return points

#----98 points from shangtang
first_three_index = [60, 72, 54]  #--left eye point, right eye point, nose's point
first_three_index = [0, 32, 85] 
first_three_index = [0, 32, 85]  #--left eye point, right eye point, nose's point
vec = [[60, 72, 54], [0, 32, 85], [0, 32, 85]]

if __name__=="__main__":
    image_name = "7_Cheering_Cheering_7_81.jpg"
    pts_name = "7_Cheering_Cheering_7_81.pts"

    points = read_pts(pts_name)
    points_clone  = points.copy()
    a_image = cv2.imread(image_name)
    h, w, _ = a_image.shape
      
    angle = 20
      
    #------cv2------
    M = cv2.getRotationMatrix2D((w/2., h/2.), angle*1., 1.0) #--以图像中心进行旋转
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = h * abs(sin) + w * abs(cos)
    b_h = h * abs(cos) + w * abs(sin)
    M[0, 2] += ((b_w / 2.) - w/2.)
    M[1, 2] += ((b_h / 2.) - h/2.)
    a_image_cv = a_image.copy()
    a_image_cv = cv2.warpAffine(a_image_cv, M, (int(w), int(h)))
    print("a_image.shape:\t", a_image.shape)
    print("a_image_cv.shape:\t", a_image_cv.shape)

    #----旋转关键点----
    #1.先去掉平移
    offset = np.array([w/2., h/2.])
    points = points - offset
    #2.进行旋转
    M = M[:,:2].T
    print(M)
    #print(M)
    points_cv = np.dot(points, M)
    #3.加上平移
    new_offset = np.asarray([b_w/2., b_h/2.])
    points_cv = points_cv + new_offset

    for i in range(98):
        x = points_cv[i][0]
        y = points_cv[i][1]
        x = int(x); y = int(y)
        color = (255, 255, 255)
        cv2.circle(a_image_cv, (x, y), 1, color, -1)
    cv2.imwrite("hh.jpg", a_image_cv)
    #------cv2------

    #----now we use points_cv and points, to get M
    #(point - offset) * M + new_offset = point_cv
    #(point - offset) * M = point_cv - new_offset
    #X = point - offset; Y = point_cv - new_offset
    #M = (X^TX)^-1 X^T Y
    points = points_clone - offset      
    points_cv = points_cv - new_offset
    points = torch.tensor(points) 
    points_cv = torch.tensor(points_cv)
    
    points_t = points.t()
    xtx = torch.inverse(torch.mm(points_t, points))
    M = torch.mm(torch.mm(xtx, points_t),points_cv)
    print(M)

    #-------batch_size of pytorch----
    points = points_clone - offset
    points = torch.tensor(points).unsqueeze(0)
    points_cv = torch.tensor(points_cv).unsqueeze(0)

    points_t = points.permute(0, 2, 1)
    xtx = torch.inverse(torch.bmm(points_t, points))
    M = torch.bmm(torch.bmm(xtx, points_t),points_cv)
    print(M)
