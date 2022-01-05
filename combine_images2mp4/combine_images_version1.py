import os
import cv2
import numpy as np

length = 1152
original_dir = "originla_plot_images_dir"

height = 1280
width = 720
#fourcc = cv2.VideoWriter_fourcc(*'MPEG')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter("original_detect.mp4", fourcc, 20.0, (width, height)) #--(width, height)

for i in range(120):
  a_image_name = "pinghua_ceshi_frame" + ("%05d" % i) + ".jpg"
  a_image_path = os.path.join(original_dir, a_image_name)
  print(a_image_path)
  a_image = cv2.imread(a_image_path)
  video.write(a_image)

cv2.destroyAllWindows()
video.release()
