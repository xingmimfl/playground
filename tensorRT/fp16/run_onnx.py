import os
import sys
import onnxruntime
import numpy as np
import glob
import cv2

def load_pts(a_pts_path):
    points = []
    for a_line in open(a_pts_path):
        a_line = a_line.strip()
        array = a_line.split()
        array = [float(x) for x in array]
        points.append(array)
    points = np.asarray(points)
    return points

def bgr2y(image):
    yuv = image.copy().astype(np.float32)
    yuv[:,:,0] = 0.299*image[:,:,2] + 0.587*image[:,:,1] + 0.114*image[:,:,0]
    yuv[:,:,1] = 0.492*(image[:,:,0] - yuv[:,:,0]) + 128
    yuv[:,:,2] = 0.877*(image[:,:,2] - yuv[:,:,0]) + 128
    yuv = np.clip(yuv, 0, 255)
    y = yuv[:, :, 0][:, :, np.newaxis]
    return y

def crop_image_f(img, points):
    img_height, img_width, _ = img.shape
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    orig_x1 = int(x1); orig_y1 = int(y1)
    orig_x2 = int(x2); orig_y2 = int(y2)
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    box_max = max(box_w, box_h) #---find max length
    box_max = box_max * 1.3 #---max_length * 1.3
    half_box_max = box_max / 2.0
    #box_w_max = box_w * 1.3
    #box_h_max = box_h * 1.3
    #half_box_w_max = box_w_max / 2.0
    #half_box_h_max = box_h_max / 2.0
    #x1 = int(cx - half_box_w_max)
    #x2 = int(cx + half_box_w_max)
    #y1 = int(cy - half_box_h_max)
    #y2 = int(cy + half_box_h_max)
    x1 = int(cx - half_box_max)
    x2 = int(cx + half_box_max)
    y1 = int(cy - half_box_max)
    y2 = int(cy + half_box_max)
    delta_x1 = 0; delta_y1 = 0
    delta_x2 = 0; delta_y2 = 0
    if x1 < 0: delta_x1 = -x1
    if y1 < 0: delta_y1 = -y1
    if (x2 > img_width -1): delta_x2 = x2 - (img_width - 1)
    if (y2 > img_height - 1): delta_y2 = y2 - (img_height - 1)
    crop_width = x2 - x1 + 1
    crop_height = y2 - y1 + 1
    crop_image = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    crop_image[delta_y1: crop_height-delta_y2, delta_x1: crop_width-delta_x2] = img[ y1 + delta_y1: y2+1 - delta_y2, x1 + delta_x1: x2+1 - delta_x2].copy()
    return crop_image, [x1, y1, x2, y2]

sess = onnxruntime.InferenceSession('landmark_lite_model_0804.onnx', None)

if __name__=="__main__":
    images_dir = ""
    target_images_dir = "WFLW_predict_onnx"
    if not os.path.exists(target_images_dir):
        os.makedirs(target_images_dir)

    count = 0
    for a_image_path in glob.iglob(images_dir + "/*.jpg"):
        print(a_image_path)
        a_image_name = os.path.basename(a_image_path)

        a_pts_path = a_image_path[:-4] + ".pts"
        points = load_pts(a_pts_path)

        a_image = cv2.imread(a_image_path)
        a_image_copy = a_image.copy()
        crop_image, crop_cor = crop_image_f(a_image, points)
        crop_h, crop_w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, (128, 128))

        height, width, _ = a_image_copy.shape
        a_image_copy = cv2.resize(a_image_copy, (4*width, 4*height))

        crop_image_y = bgr2y(crop_image)
        crop_image_y = crop_image_y.transpose(2, 0, 1)

        crop_image_y = np.expand_dims(crop_image_y, axis=0) / 255.0 * 4
        print("crop_image_y.size:\t", crop_image_y.shape)

        landmarks_out = sess.run(None, {'input':crop_image_y})
        landmarks_out = landmarks_out[0]
        print(landmarks_out)
        print("landmarks_out.shape:\t", landmarks_out.shape)
        landmarks_out = landmarks_out.squeeze() / 64.0
        last_points = np.zeros((98, 2))
        lines_vec = []
        for i in range(98):
            x = landmarks_out[2*i] * crop_w + crop_cor[0]
            y = landmarks_out[2*i+1] * crop_h + crop_cor[1]
            a_line = str(x) + " " + str(y) + "\n"
            lines_vec.append(a_line)
            x = 4 * x; y = 4 * y
            color = (0, 255, 0)
            cv2.circle(a_image_copy, (int(x), int(y)), 3, color, -1)

        a_target_image_path = os.path.join(target_images_dir, a_image_name)
        cv2.imwrite(a_target_image_path, a_image_copy)

        a_target_pts_path = os.path.join(target_images_dir, a_image_name)[:-4] + ".pts"
        f = open(a_target_pts_path, "w")
        f.writelines(lines_vec)
        f.close()
