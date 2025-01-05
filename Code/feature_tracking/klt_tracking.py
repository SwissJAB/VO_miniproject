import cv2
import numpy as np
import os

def track_keypoints(prev_frame, curr_frame, prev_keypoints, landmarks, lk_params):
    """
    Tracks keypoints from the previous frame to the current frame using KLT optical flow.
    
    Args:
        prev_frame (ndarray): Grayscale image of the previous frame.
        curr_frame (ndarray): Grayscale image of the current frame.
        prev_keypoints (ndarray): Array of keypoints from the previous frame.
        landmarks (ndarray): Corresponding 3D landmarks.
        
    Returns:
        valid_prev_keypoints (ndarray): Valid keypoints from the previous frame.
        valid_curr_keypoints (ndarray): Valid tracked keypoints in the current frame.
        associated_landmarks (ndarray): Corresponding 3D landmarks.
    """
    # Calculate optical flow
    curr_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_keypoints, None, **lk_params)
    
    if curr_keypoints is None or status is None:
        print("Optical flow failed. Returning previous keypoints.")
        return prev_keypoints, prev_keypoints, None, None
    
    # Filter keypoints based on tracking status
    valid_prev_keypoints = prev_keypoints[status.flatten() == 1]
    valid_curr_keypoints = curr_keypoints[status.flatten() == 1]
    associated_landmarks = landmarks[status.flatten() == 1, :]  # Corresponding 3D landmarks
    
    return valid_prev_keypoints, valid_curr_keypoints, associated_landmarks