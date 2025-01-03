import numpy as np


def describeKeypoints(img, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    #pass
    N = keypoints.shape[1]
    print("N:", N)
    desciptors = np.zeros([(2*r+1)**2, N])
    print("desciptors shape:", desciptors.shape)
    padded = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)
    print("padded shape:", padded.shape)

    for i in range(N):
        kp = keypoints[:, i].astype(int) + r
        print("kp:", kp)
        # Check if the slice is within bounds
        x_start, x_end = kp[0] - r, kp[0] + r + 1
        y_start, y_end = kp[1] - r, kp[1] + r + 1
        if x_start < 0 or x_end > padded.shape[0] or y_start < 0 or y_end > padded.shape[1]:
            print(f"Skipping keypoint {kp - r} due to out-of-bounds patch.")
            continue
        desciptors[:, i] = padded[x_start:x_end, y_start:y_end].flatten()
    return desciptors


