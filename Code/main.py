import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from descriptor_utils import get_descriptors
from get_descriptors.match_descriptors import matchDescriptors
from get_descriptors.plot_matches import plotMatches

### IMPORT THE DATA ###

data_set_root_file = './Datasets/'
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
    
# plt.clf()
# plt.close()
# plt.imshow(img2, cmap='gray')
# plt.plot(keypoints2[1, :], keypoints2[0, :], 'rx', linewidth=2)
# plotMatches(matches, keypoints2, keypoints1)
# plt.tight_layout()
# plt.axis('off')
# plt.show()

# Extract matched keypoints
matched_keypoints1 = keypoints1[:, matches != -1]
matched_keypoints2 = keypoints2[:, matches[matches != -1]]



# Compute the Essential Matrix
print(data_set_root_file + dataset_curr + '/K.txt')
K = np.genfromtxt(data_set_root_file + dataset_curr + '/K.txt', delimiter=',', dtype=float).reshape(3, 3)
print(K)
E, mask = cv2.findEssentialMat(matched_keypoints1.T, matched_keypoints2.T, K)

# Decompose the Essential Matrix to obtain rotation and translation
_, R, t, mask = cv2.recoverPose(E, matched_keypoints1.T, matched_keypoints2.T, K)

print("Rotation matrix:")
print(R)
print("Translation vector:")
print(t)

# Triangulate points to get 3D coordinates
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))
points_4d_hom = cv2.triangulatePoints(K @ P1, K @ P2, matched_keypoints1, matched_keypoints2)
points_3d = points_4d_hom[:3] / points_4d_hom[3]

# Initialize pose
pose = np.eye(4)

# Update pose with the new rotation and translation
pose[:3, :3] = R @ pose[:3, :3]
pose[:3, 3] += t.flatten()

# Plot the 3D keypoints and camera poses
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D keypoints
ax.scatter(points_3d[0], points_3d[1], points_3d[2], c='b', marker='o')

# Plot the camera poses
ax.scatter(0, 0, 0, c='r', marker='^')  # Initial camera pose
ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3], c='g', marker='^')  # New camera pose

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Keypoints and Camera Poses')

plt.show()