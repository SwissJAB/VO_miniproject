from two_view_geometry.fundamental_eight_point_normalized import fundamentalEightPointNormalized

def estimateEssentialMatrix(p1, p2, K1, K2):
    """ estimates the essential matrix given matching point coordinates,
        and the camera calibration K

     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2
      - K1 np.ndarray(3,3): calibration matrix of camera 1
      - K2 np.ndarray(3,3): calibration matrix of camera 2

     Output:
      - E np.ndarray(3,3) : fundamental matrix
    """
    pass

    F = fundamentalEightPointNormalized(p1, p2)
    print("F:\n", F)

    # Compute the essential matrix from the fundamental matrix given K
    E = K2.T @ F @ K1

    return E
