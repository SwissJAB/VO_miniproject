import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
from descriptor_utils import get_descriptors
from get_descriptors.match_descriptors import matchDescriptors
from get_descriptors.plot_matches import plotMatches

from two_view_geometry.estimate_essential_matrix import estimateEssentialMatrix
from two_view_geometry.decompose_essential_matrix import decomposeEssentialMatrix
from two_view_geometry.disambiguate_relative_pose import disambiguateRelativePose
from two_view_geometry.linear_triangulation import linearTriangulation
from two_view_geometry.draw_camera import drawCamera

# from get_descriptors.SIFT.compute_blurred_images import computeBlurredImages
# from get_descriptors.SIFT.compute_descriptors import computeDescriptors 
# from get_descriptors.SIFT.compute_difference_of_gaussians import computeDifferenceOfGaussians 
# from get_descriptors.SIFT.compute_image_pyramid import computeImagePyramid 
# from get_descriptors.SIFT.extract_keypoints import extractKeypoints

from feature_tracking.klt_tracking import track_keypoints

### OPEN CONFIG FILE ###
with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

### IMPORT THE DATA ###
data_set_root_file = config['DATA']['rootdir']
datasets = config['DATA']['datasets']
dataset_curr = config['DATA']['curr_dataset']
assert dataset_curr in datasets, f"Invalid dataset: {dataset_curr}"


### SELECT 2 IMAGES ###
images = ['img_00000.png','img_00050.png']

# Load the images
img1 = cv2.imread(f'{data_set_root_file}{datasets[0]}/images/{images[0]}', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(f'{data_set_root_file}{datasets[0]}/images/{images[1]}', cv2.IMREAD_GRAYSCALE)

descriptor='shi_tomasi'

if descriptor == 'harris' or descriptor == 'shi_tomasi':
    ### COMPUTE CORNER SCORES ###
    corner_patch_size = 9
    harris_kappa = 0.08
    num_keypoints = 200
    nonmaximum_supression_radius = 8
    descriptor_radius = 9
    match_lambda = 4

    descriptors1, keypoints1 = get_descriptors(img1, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius)

    descriptors2, keypoints2 = get_descriptors(img2, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius)

    matches = matchDescriptors(descriptors2, descriptors1, match_lambda)
    print(f"Number of matches: {np.sum(matches != -1)}")
    
    # Plot the matches
    # plotMatches(matches, keypoints2, keypoints1)


elif descriptor == 'sift':
    # User parameters
    rotation_invariant =True       # Enable rotation invariant SIFT
    rotation_img2_deg = 60          # Rotate the second image to be matched

    # sift parameters
    contrast_threshold = 0.04       # for feature matching
    sift_sigma = 1.0                # sigma used for blurring
    rescale_factor = 0.3            # rescale images to make it faster
    num_scales = 3                  # number of scales per octave
    num_octaves = 5                 # number of octaves
    match_lambda = 4

    imgs = [img1, img2]
    keypoint_locations = []
    keypoint_descriptors = []

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    print(f"Number of keypoints, descriptors in image 1: {len(keypoints1), len(descriptors1)}")
    print(f"Number of keypoints, descriptors in image 2: {len(keypoints2), len(descriptors2)}")
    matches = matchDescriptors(descriptors2, descriptors1, match_lambda)
    print(f"Number of matches: {np.sum(matches != -1)}")
else:
    raise ValueError("Invalid descriptor type")

# Extract matched keypoints
query_indices = np.nonzero(matches >= 0)[0]
match_indices = matches[query_indices]

matched_keypoints1 = keypoints1[:, match_indices]
matched_keypoints2 = keypoints2[:, query_indices]

# Convert matched keypoints to homogeneous coordinates
matched_keypoints1 = np.r_[matched_keypoints1, np.ones((1, matched_keypoints1.shape[1]))]
matched_keypoints2 = np.r_[matched_keypoints2, np.ones((1, matched_keypoints2.shape[1]))]

# Switch the coordinates of matched keypoints. the second column of matched_keypoints1 will be the first column and vice versa
matched_keypoints1 = np.array([matched_keypoints1[1], matched_keypoints1[0], matched_keypoints1[2]])
matched_keypoints2 = np.array([matched_keypoints2[1], matched_keypoints2[0], matched_keypoints2[2]])

print(matched_keypoints1)
# Compute the Essential Matrix
print("Path:", data_set_root_file + dataset_curr + '/K.txt')
K = np.genfromtxt(data_set_root_file + dataset_curr + '/K.txt', delimiter=',', dtype=float).reshape(3, 3)
print("K:", K)

print("matched_keypoints1:", matched_keypoints1.shape)
print("matched_keypoints2:", matched_keypoints2.shape)
E, mask = cv2.findEssentialMat(matched_keypoints1[:2].T, matched_keypoints2[:2].T, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
print("Mask shape:", mask.shape)

matched_keypoints1 = matched_keypoints1[:, mask.ravel() == 1] 
matched_keypoints2 = matched_keypoints2[:, mask.ravel() == 1]



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
ax.scatter(P[0,:], P[1,:], P[2,:], marker = 'o')

# Display camera pose
drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale = 2)
ax.text(-0.1,-0.1,-0.1,"Cam 1")

center_cam2_W = -R.T @ t

drawCamera(ax, center_cam2_W, R.T, length_scale = 2)
ax.text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1,'Cam 2')

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display matched points
ax = fig.add_subplot(1,3,2)
ax.imshow(img1)
ax.scatter(matched_keypoints1[0,:], matched_keypoints1[1,:], color = 'y', marker='s')
for i in range(matched_keypoints1.shape[1]):
    ax.plot([matched_keypoints1[0,i], matched_keypoints2[0,i]], [matched_keypoints1[1,i], matched_keypoints2[1,i]], 'r-')
ax.set_title("Image 1")

ax = fig.add_subplot(1,3,3)
ax.imshow(img2)
ax.scatter(matched_keypoints2[0,:], matched_keypoints2[1,:], color = 'y', marker='s')
# for i in range(matched_keypoints2.shape[1]):
#     ax.plot([matched_keypoints1[1,i], matched_keypoints2[1,i]], [matched_keypoints1[0,i], matched_keypoints2[0,i]], 'r-')
ax.set_title("Image 2")

plt.show()