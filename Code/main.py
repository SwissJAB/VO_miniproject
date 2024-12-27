import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from descriptor_utils import get_descriptors
from get_descriptors.match_descriptors import matchDescriptors
from get_descriptors.plot_matches import plotMatches

from two_view_geometry.estimate_essential_matrix import estimateEssentialMatrix
from two_view_geometry.decompose_essential_matrix import decomposeEssentialMatrix
from two_view_geometry.disambiguate_relative_pose import disambiguateRelativePose
from two_view_geometry.linear_triangulation import linearTriangulation
from two_view_geometry.draw_camera import drawCamera

### IMPORT THE DATA ###

data_set_root_file = '../Datasets/'
datasets = ['parking','kitti','malaga']
dataset_curr = datasets[0]

### SELECT 2 IMAGES ###
images = ['img_00000.png','img_00001.png']

# Load the images
img1 = cv2.imread(f'{data_set_root_file}{datasets[0]}/images/{images[0]}', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f'{data_set_root_file}{datasets[0]}/images/{images[1]}', cv2.IMREAD_GRAYSCALE)


### COMPUTE CORNER SCORES ###
corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 4

# Get descriptors for the first 2 images
start_time = time.time()
descriptors1, keypoints1 = get_descriptors(img1, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius)
end_time = time.time()
print(f"Time to get descriptors for the first image: {end_time - start_time} seconds")

start_time = time.time()
descriptors2, keypoints2 = get_descriptors(img2, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius)
end_time = time.time()
print(f"Time to get descriptors for the second image: {end_time - start_time} seconds")

matches = matchDescriptors(descriptors2, descriptors1, match_lambda)
    
plt.clf()
plt.close()
plt.imshow(img2, cmap='gray')
plt.plot(keypoints2[1, :], keypoints2[0, :], 'rx', linewidth=2)
plotMatches(matches, keypoints2, keypoints1)
plt.tight_layout()
plt.axis('off')
plt.show()

# Extract matched keypoints
matched_keypoints1 = keypoints1[:, matches != -1]
matched_keypoints2 = keypoints2[:, matches[matches != -1]]

# Convert matched keypoints to homogeneous coordinates
matched_keypoints1 = np.r_[matched_keypoints1, np.ones((1, matched_keypoints1.shape[1]))]
matched_keypoints2 = np.r_[matched_keypoints2, np.ones((1, matched_keypoints2.shape[1]))]
print(matched_keypoints1)
# Compute the Essential Matrix
print(data_set_root_file + dataset_curr + '/K.txt')
K = np.genfromtxt(data_set_root_file + dataset_curr + '/K.txt', delimiter=',', dtype=float).reshape(3, 3)
print(K)
E = estimateEssentialMatrix(matched_keypoints1, matched_keypoints2, K, K)

# Decompose the Essential Matrix to obtain rotation and translation
Rots, u3 = decomposeEssentialMatrix(E)

# Disambiguate among the four possible configurations
R, t = disambiguateRelativePose(Rots, u3, matched_keypoints1, matched_keypoints2, K, K)
print("Rotation matrix:")
print(R)
print("Translation vector:")
print(t)

# Triangulate a point cloud using the final transformation (R,T)
M1 = K @ np.eye(3,4)
M2 = K @ np.c_[R, t]
P = linearTriangulation(matched_keypoints1, matched_keypoints2, M1, M2)
# Visualize the 3-D scene
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')

# R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]

# P is a [4xN] matrix containing the triangulated point cloud (in
# homogeneous coordinates), given by the function linearTriangulation
ax.scatter(P[1,:], P[0,:], P[2,:], marker = 'o')

# Display camera pose
drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale = 2)
ax.text(-0.1,-0.1,-0.1,"Cam 1")

center_cam2_W = -R.T @ t
drawCamera(ax, center_cam2_W, R.T, length_scale = 2)
ax.text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1,'Cam 2')

# Display matched points
ax = fig.add_subplot(1,3,2)
ax.imshow(img1)
ax.scatter(matched_keypoints1[1,:], matched_keypoints1[0,:], color = 'y', marker='s')
ax.set_title("Image 1")

ax = fig.add_subplot(1,3,3)
ax.imshow(img2)
ax.scatter(matched_keypoints2[1,:], matched_keypoints2[0,:], color = 'y', marker='s')
ax.set_title("Image 2")

plt.show()