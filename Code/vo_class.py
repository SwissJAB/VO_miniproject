import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
import os

from descriptor_utils import get_descriptors_harris, get_descriptors_st
from get_descriptors.match_descriptors import matchDescriptors
from get_descriptors.plot_matches import plotMatches

from two_view_geometry.estimate_essential_matrix import estimateEssentialMatrix
from two_view_geometry.decompose_essential_matrix import decomposeEssentialMatrix
from two_view_geometry.disambiguate_relative_pose import disambiguateRelativePose
from two_view_geometry.linear_triangulation import linearTriangulation
from two_view_geometry.draw_camera import drawCamera

from feature_tracking.klt_tracking import track_keypoints


class VisualOdometryPipeline:
    def __init__(self, config_path='Code/config.yaml'):
        """
        Constructor loads and parses the configuration file, reads the necessary images,
        and initializes internal variables.
        """
        # Load YAML config
        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        
        # Set up data directories
        self.data_rootdir = os.getcwd() + self.config['DATA']['rootdir']
        self.datasets = self.config['DATA']['datasets']
        self.dataset_curr = self.config['DATA']['curr_dataset']
        assert self.dataset_curr in self.datasets, f"Invalid dataset: {self.dataset_curr}"

        # Load images
        self.img1 = cv2.imread(f'{self.data_rootdir}{self.dataset_curr}{self.config["DATA"]["init_img_1"]}', cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread(f'{self.data_rootdir}{self.dataset_curr}{self.config["DATA"]["init_img_2"]}', cv2.IMREAD_GRAYSCALE)
        

        # Read the camera calibration
        K_path = self.data_rootdir + self.dataset_curr + '/K.txt'
        self.K = np.genfromtxt(K_path, delimiter=',', dtype=float).reshape(3, 3)

        # Chosen descriptor from config
        self.curr_desc = self.config['FEATURES']['curr_detector']
        self.descriptor_name = self.config['FEATURES']['detectors'][self.curr_desc]

        # Placeholders for keypoints, descriptors, matches
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.matches = None
        self.matched_keypoints1 = None
        self.matched_keypoints2 = None
        self.E = None
        self.mask = None
        self.R = None
        self.t = None
        self.P = None

    def run(self):
        """
        Execute the entire visual odometry pipeline in order:
          1. Detect & compute descriptors (Harris, Shi-Tomasi, or SIFT)
          2. Match descriptors
          3. Compute essential matrix with RANSAC
          4. Decompose E into R, t
          5. Triangulate
          6. Visualize results
        """
        self._detect_and_compute()
        self._match_descriptors()
        self._find_essential_matrix()
        self._decompose_E()
        self._triangulate()
        self._visualize()

    def _detect_and_compute(self):
        """
        Detect keypoints and compute descriptors based on self.descriptor_name.
        """
        print(f"Descriptor: {self.descriptor_name}")
        
        if self.descriptor_name == 'harris':
            self._detect_harris()
        elif self.descriptor_name == 'shi_tomasi':
            self._detect_shi_tomasi()
        elif self.descriptor_name == 'sift':
            self._detect_sift()
        else:
            raise ValueError("Invalid descriptor type")

    def _detect_harris(self):
        print("Harris")
        # For speed, read parameters only once into local variables
        harris_cfg = self.config['HARRIS']
        self.descriptors1, self.keypoints1 = get_descriptors_harris(
            self.img1,
            harris_cfg['corner_patch_size'], 
            harris_cfg['kappa'], 
            harris_cfg['num_keypoints'], 
            harris_cfg['nonmaximum_supression_radius'], 
            harris_cfg['descriptor_radius']
        )

        self.descriptors2, self.keypoints2 = get_descriptors_harris(
            self.img2,
            harris_cfg['corner_patch_size'], 
            harris_cfg['kappa'],
            harris_cfg['num_keypoints'], 
            harris_cfg['nonmaximum_supression_radius'],
            harris_cfg['descriptor_radius']
        )

    def _detect_shi_tomasi(self):
        st_cfg = self.config['SHI_TOMASI']
        self.descriptors1, self.keypoints1 = get_descriptors_st(
            self.img1,
            st_cfg['corner_patch_size'], 
            st_cfg['num_keypoints'],
            st_cfg['nonmaximum_supression_radius'], 
            st_cfg['descriptor_radius']
        )
        self.descriptors2, self.keypoints2 = get_descriptors_st(
            self.img2,
            st_cfg['corner_patch_size'], 
            st_cfg['num_keypoints'],
            st_cfg['nonmaximum_supression_radius'], 
            st_cfg['descriptor_radius']
        )

    def _detect_sift(self):
        sift_cfg = self.config['SIFT']
        sift = cv2.SIFT_create(
            nfeatures=sift_cfg['nfeatures'],
            contrastThreshold=sift_cfg['contrast_threshold'],
            sigma=sift_cfg['sigma'],
            nOctaveLayers=sift_cfg['n_otave_layers']
        )
        kp1, desc1 = sift.detectAndCompute(self.img1, None)
        kp2, desc2 = sift.detectAndCompute(self.img2, None)

        # Convert to np arrays
        keypoints1_np = np.array([kp.pt for kp in kp1]).T  # shape (2, N1)
        keypoints2_np = np.array([kp.pt for kp in kp2]).T  # shape (2, N2)
        descriptors1_t = desc1.T                            # shape (128, N1)
        descriptors2_t = desc2.T                            # shape (128, N2)

        self.keypoints1 = np.array([keypoints1_np[1], keypoints1_np[0]])  # swapped to match other things, not great TODO: fix
        self.keypoints2 = np.array([keypoints2_np[1], keypoints2_np[0]])
        self.descriptors1 = descriptors1_t
        self.descriptors2 = descriptors2_t

    def _match_descriptors(self):
        """
        Use matchDescriptors() from local library.
        Then extract matched keypoints in homogeneous coordinates.
        """
        if self.descriptor_name in ['harris', 'shi_tomasi']:
            match_lambda = self.config[self.descriptor_name.upper()]['match_lambda']
        else:
            match_lambda = self.config['SIFT']['match_lambda']

        # matches is a 1D array with length = # of query descriptors
        self.matches = matchDescriptors(
            self.descriptors2,  # Query
            self.descriptors1,  # Database
            match_lambda
        )
        print(f"Number of matches: {np.sum(self.matches != -1)}")

        query_indices = np.nonzero(self.matches >= 0)[0]
        match_indices = self.matches[query_indices]
        print("query_indices shape:", query_indices.shape)
        print("match_indices shape:", match_indices.shape)

        matched_keypoints1 = self.keypoints1[:, match_indices]
        matched_keypoints2 = self.keypoints2[:, query_indices]

        # Convert to homogeneous coords
        matched_keypoints1 = np.r_[matched_keypoints1, np.ones((1, matched_keypoints1.shape[1]))]
        matched_keypoints2 = np.r_[matched_keypoints2, np.ones((1, matched_keypoints2.shape[1]))]

        # Switch the coordinates if truly need them swapped again TODO: This is not great yet
        self.matched_keypoints1 = np.array([matched_keypoints1[1], matched_keypoints1[0], matched_keypoints1[2]])
        self.matched_keypoints2 = np.array([matched_keypoints2[1], matched_keypoints2[0], matched_keypoints2[2]])
        print("matched_keypoints1:\n", self.matched_keypoints1)

    def _find_essential_matrix(self):
        """
        Find the essential matrix using OpenCV's RANSAC-based findEssentialMat.
        """
        print("Path:", os.path.join(self.data_rootdir, self.dataset_curr, 'K.txt'))
        print("K:", self.K)
        print("K shape:", self.K.shape)

        print("matched_keypoints1:", self.matched_keypoints1.shape)
        print("matched_keypoints2:", self.matched_keypoints2.shape)

        E, mask = cv2.findEssentialMat(
            self.matched_keypoints1[:2].T,  # shape => Nx2
            self.matched_keypoints2[:2].T,
            self.K, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )
        print("Mask shape:", mask.shape if mask is not None else "No mask")
        
        self.E = E
        self.mask = mask.ravel() == 1 if mask is not None else None

        # Filter matched keypoints by inlier mask
        if self.mask is not None:
            self.matched_keypoints1 = self.matched_keypoints1[:, self.mask]
            self.matched_keypoints2 = self.matched_keypoints2[:, self.mask]

    def _decompose_E(self):
        """
        Decompose the essential matrix into R, t.
        Disambiguate among the four possible solutions.
        """
        Rots, u3 = decomposeEssentialMatrix(self.E)
        R, t = disambiguateRelativePose(
            Rots, u3, 
            self.matched_keypoints1, 
            self.matched_keypoints2, 
            self.K, self.K
        )
        self.R = R
        self.t = t
        print("Rotation matrix:")
        print(self.R)
        print("Translation vector:")
        print(self.t)

    def _triangulate(self):
        """
        Triangulate matched points into 3D using linearTriangulation.
        """
        M1 = self.K @ np.eye(3, 4)
        M2 = self.K @ np.c_[self.R, self.t]
        self.P = linearTriangulation(
            self.matched_keypoints1,
            self.matched_keypoints2, 
            M1, 
            M2
        )

    def _visualize(self):
        """
        Create a 3D plot of the triangulated points and show matched features on the images.
        """
        fig = plt.figure()

        # 3D plot (point cloud + cameras)
        ax_3d = fig.add_subplot(1, 3, 1, projection='3d')
        ax_3d.scatter(self.P[0, :], self.P[1, :], self.P[2, :], marker='o')

        # Display camera 1
        drawCamera(ax_3d, np.zeros((3,)), np.eye(3), length_scale=2)
        ax_3d.text(-0.1, -0.1, -0.1, "Cam 1")

        # Display camera 2
        center_cam2_W = -self.R.T @ self.t
        drawCamera(ax_3d, center_cam2_W, self.R.T, length_scale=2)
        ax_3d.text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1, 'Cam 2')

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')

        # 2D plots with matched points
        ax_img1 = fig.add_subplot(1, 3, 2)
        ax_img1.imshow(self.img1, cmap='gray')
        ax_img1.scatter(self.matched_keypoints1[0, :], self.matched_keypoints1[1, :],
                        color='y', marker='s')
        for i in range(self.matched_keypoints1.shape[1]):
            ax_img1.plot(
                [self.matched_keypoints1[0, i], self.matched_keypoints2[0, i]],
                [self.matched_keypoints1[1, i], self.matched_keypoints2[1, i]],
                'r-'
            )
        ax_img1.set_title("Image 1")

        ax_img2 = fig.add_subplot(1, 3, 3)
        ax_img2.imshow(self.img2, cmap='gray')
        ax_img2.scatter(self.matched_keypoints2[0, :], self.matched_keypoints2[1, :],
                        color='y', marker='s')
        ax_img2.set_title("Image 2")

        plt.show()


if __name__ == "__main__":
    pipeline = VisualOdometryPipeline(config_path='Code/config.yaml')
    pipeline.run()
