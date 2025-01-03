import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from get_descriptors.harris import harris
from get_descriptors.shi_tomasi import shi_tomasi
from get_descriptors.select_keypoints import selectKeypoints
from get_descriptors.describe_keypoints import describeKeypoints


def get_descriptors_harris(image, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius):

    harris_scores = harris(image, corner_patch_size, harris_kappa)
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    return descriptors, keypoints

def get_descriptors_harris_cv2(image, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius):
    harris_scores = cv2.cornerHarris(image, corner_patch_size, 3, harris_kappa)
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    return descriptors, keypoints

def get_descriptors_st(image, corner_patch_size, num_keypoints, nonmaximum_supression_radius, descriptor_radius):
    harris_scores = shi_tomasi(image, corner_patch_size)
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    return descriptors, keypoints

def get_descriptors_st_cv2(image, corner_patch_size, num_keypoints, nonmaximum_supression_radius, descriptor_radius):
    keypoints = cv2.goodFeaturesToTrack(image, num_keypoints, 0.01, nonmaximum_supression_radius)
    keypoints = np.squeeze(keypoints.T)
    print(keypoints.shape)
    print("done")
    #keypoints = selectKeypoints(keypoints, num_keypoints, nonmaximum_supression_radius)
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    return descriptors, keypoints