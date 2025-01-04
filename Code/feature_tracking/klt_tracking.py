import cv2
import numpy as np
import os

def track_keypoints(prev_frame, curr_frame, prev_keypoints, landmarks):
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
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    # Calculate optical flow
    print("previous frame shape:", prev_frame.shape)
    print("curr shape:", curr_frame.shape)
    curr_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_keypoints, None, **lk_params)
    
    if curr_keypoints is None or status is None:
        print("Optical flow failed. Returning previous keypoints.")
        return prev_keypoints, prev_keypoints, None, None
    
    # Filter keypoints based on tracking status
    valid_prev_keypoints = prev_keypoints[status.flatten() == 1]
    valid_curr_keypoints = curr_keypoints[status.flatten() == 1]
    associated_landmarks = landmarks[status.flatten() == 1, :]  # Corresponding 3D landmarks
    
    return valid_prev_keypoints, valid_curr_keypoints, associated_landmarks

def draw_keypoints(frame, keypoints, color=(0, 255, 0)):
    """
    Draw keypoints on an image.
    
    Args:
        frame (ndarray): The image on which to draw keypoints.
        keypoints (ndarray): Array of keypoints.
        color (tuple): Color of the keypoints.
    
    Returns:
        frame (ndarray): Image with keypoints drawn.
    """
    for point in keypoints:
        x, y = point.ravel()
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return frame

def load_images_from_folder(folder):
    """
    Load all images from a given folder.
    
    Args:
        folder (str): Path to the folder containing images.
        
    Returns:
        images (list): List of loaded images.
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

if __name__ == '__main__':
    dataset_path = './Datasets/parking/images'
    images = load_images_from_folder(dataset_path)
    
    if len(images) < 2:
        print("Not enough images in the dataset.")
        exit()
    
    prev_frame = images[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect initial keypoints using Shi-Tomasi Corner Detector
    prev_keypoints = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    if prev_keypoints is None:
        print("No keypoints detected in the initial frame.")
        exit()
    
    for i in range(1, len(images)):
        curr_frame = images[i]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        valid_prev_keypoints, valid_curr_keypoints, status, err = track_keypoints(prev_gray, curr_gray, prev_keypoints)
        
        if valid_curr_keypoints is None or len(valid_curr_keypoints) == 0:
            print("No valid keypoints found. Re-detecting keypoints.")
            prev_keypoints = cv2.goodFeaturesToTrack(curr_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            if prev_keypoints is None:
                print("Failed to re-detect keypoints. Exiting.")
                break
            prev_gray = curr_gray.copy()
            continue
        
        # Draw keypoints
        frame_with_keypoints = draw_keypoints(curr_frame, valid_curr_keypoints)
        cv2.imshow('Tracked Keypoints', frame_with_keypoints)
        
        # Update previous frame and keypoints
        prev_gray = curr_gray.copy()
        prev_keypoints = valid_curr_keypoints.reshape(-1, 1, 2)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
