import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from get_descriptors.harris import harris
from get_descriptors.shi_tomasi import shi_tomasi
from get_descriptors.select_keypoints import selectKeypoints
from get_descriptors.describe_keypoints import describeKeypoints

def get_descriptors(image, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius, descriptor='harris'):
    if descriptor == 'harris':
        start_time = time.time()
        harris_scores = harris(image, corner_patch_size, harris_kappa)
        time = time.time() - start_time
        print(f"Harris computation time: {time:.4f} seconds")
        # in txt file save the time for shi-tomasi
        with open("Code/plots/harris_time.txt", "w") as f:
            f.write(str(time))
    elif descriptor == 'shi tomasi':
        start_time = time.time()
        harris_scores = shi_tomasi(image, corner_patch_size)
        time = time.time() - start_time
        print(f"Shi-Tomasi computation time: {time:.4f} seconds")
        with open("Code/plots/shi_tomasi_time.txt", "w") as f:
            f.write(str(time))
    else:
        raise ValueError(f"Invalid descriptor type: {descriptor}")
    start_time = time.time()
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
    keypoints_time = time.time() - start_time
    print(f"Keypoints computation time: {keypoints_time:.4f} seconds")
    start_time = time.time()
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    descriptors_time = time.time() - start_time
    print(f"Descriptors computation time: {descriptors_time:.4f} seconds")

    return descriptors, keypoints


def get_descriptors_harris(image, corner_patch_size, harris_kappa, num_keypoints, nonmaximum_supression_radius, descriptor_radius):
    start_time = time.time()
    harris_scores = harris(image, corner_patch_size, harris_kappa)
    harris_time = time.time() - start_time
    print(f"Harris computation time: {harris_time:.4f} seconds")
    with open("Code/plots/harris_time.txt", "w") as f:
        f.write(str(harris_time))
    start_time = time.time()
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
    keypoints_time = time.time() - start_time
    print(f"Keypoints computation time: {keypoints_time:.4f} seconds")
    start_time = time.time()
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    descriptors_time = time.time() - start_time
    print(f"Descriptors computation time: {descriptors_time:.4f} seconds")
    return descriptors, keypoints


def get_descriptors_st(image, corner_patch_size, num_keypoints, nonmaximum_supression_radius, descriptor_radius):
    start_time = time.time()
    harris_scores = shi_tomasi(image, corner_patch_size)
    harris_time = time.time() - start_time
    print(f"Shi-Tomasi computation time: {harris_time:.4f} seconds")
    with open("Code/plots/shi_tomasi_time.txt", "w") as f:
        f.write(str(harris_time))
    start_time = time.time()
    keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
    keypoints_time = time.time() - start_time
    print(f"Keypoints computation time: {keypoints_time:.4f} seconds")
    start_time = time.time()
    descriptors = describeKeypoints(image, keypoints, descriptor_radius)
    descriptors_time = time.time() - start_time
    print(f"Descriptors computation time: {descriptors_time:.4f} seconds")
    return descriptors, keypoints