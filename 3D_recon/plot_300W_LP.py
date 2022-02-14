import os
import scipy.io as sio
import cv2
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.switch_backend('agg')
import glob

def angle2matrix_3ddfa(angles):
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  sin(x)],
                 [0, -sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, -sin(y)],
                 [      0, 1,      0],
                 [sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), sin(z), 0],
                 [-sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)

file_name = "BFM.mat"
bfm = sio.loadmat(file_name)
model = bfm['model']
model = model[0,0]

# change dtype from double(np.float64) to np.float32,
# since big matrix process(espetially matrix dot) is too slow in python.
model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
model['shapePC'] = model['shapePC'].astype(np.float32)
model['shapeEV'] = model['shapeEV'].astype(np.float32)
model['expPC'] = model['expPC'].astype(np.float32)
model['expEV'] = model['expEV'].astype(np.float32)
# matlab start with 1. change to 0 in python.
model['tri'] = model['tri'].T.copy(order = 'C').astype(np.int32) - 1
model['tri_mouth'] = model['tri_mouth'].T.copy(order = 'C').astype(np.int32) - 1
# kpt ind
model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

print("model_shapePC.shape:\t", model['shapePC'].shape)
print('model_expPC.shape:\t', model['expPC'].shape)

shape_mu = model['shapeMU']
shape_pc = model['shapePC']; shape_ev = model['shapeEV']
exp_pc = model['expPC']; exp_ev = model['expEV']

if __name__=="__main__":
    images_dir = "AFW"
    target_images_dir = "plot_images_dir"
    if not os.path.exists(target_images_dir):
        os.makedirs(target_images_dir)

    for a_image_path in glob.iglob(images_dir + "/*.jpg"):
        a_mat_path = a_image_path[:-4] + ".mat"
        a_image_name = os.path.basename(a_image_path)

        a_image = cv2.imread(a_image_path)[:,:,::-1] / 255.
        h, w, _ = a_image.shape

        info = sio.loadmat(a_mat_path)
        pose_para = info['Pose_Para'].T.astype(np.float32)
        shape_para = info['Shape_Para'].astype(np.float32)
        exp_para = info['Exp_Para'].astype(np.float32)

        vertices = shape_mu + shape_pc.dot(shape_para) + exp_pc.dot(exp_para)
        vertices = vertices.reshape(-1, 3)

        s = pose_para[-1, 0]
        angles = pose_para[:3, 0]
        t = pose_para[3:6, 0]
        t = np.squeeze(np.array(t, dtype = np.float32))
        t = t[np.newaxis, :]
        angles = np.asarray([angles[0], angles[1], angles[2]])

        rotate_matrix = angle2matrix_3ddfa(angles)
        transformed_vertices = s * vertices.dot(rotate_matrix.T)
        transformed_vertices = transformed_vertices + t
        transformed_vertices[:, 1] = h - transformed_vertices[:, 1] - 1

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(a_image)
        x_vec, y_vec, z_vec = transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2]
        ax.scatter(x_vec, y_vec, color = 'r', s = 0.01)
        a_target_image_path = os.path.join(target_images_dir, a_image_name)
        fig.savefig(a_target_image_path)

