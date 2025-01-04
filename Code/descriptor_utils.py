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








def get_descriptors_st_cv2(image, corner_patch_size, num_keypoints, nonmaximum_supression_radius, descriptor_radius, quality_level, debug=True):
    keypoints = cv2.goodFeaturesToTrack(image, num_keypoints, quality_level, nonmaximum_supression_radius)
    draw_kp = cv2.drawKeypoints(image, keypoints, image)
    cv2.imshow("Keypoints with draw keyponts", draw_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("keypoints shape:", keypoints.shape)
    keypoints = keypoints.reshape(-1, 2).T # Shape is now (2, N), x, y 
    print("keypoints shape after squeeze:", keypoints.shape)
    if debug:
        # Plot the keypoints on the image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(keypoints.shape[1]):  # Loop over the second axis
            x, y = int(keypoints[0, i]), int(keypoints[1, i])  # Extract x, y
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow("Keypoints", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    return descriptors, keypoints