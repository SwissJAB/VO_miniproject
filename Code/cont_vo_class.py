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

        self.baseline_angle_thresh = np.deg2rad(self.config['CONT_VO']['baseline_angle_thresh']) # TODO: Add logic for this
        self.global_poses = []


    def initialize(self):
        # TODO: This has the image references hardcoded inside the function calls. Should be fixed
        key1, desc1, key2, desc2 = self._detect_and_compute()
        print("key1 shape, key2 shape, desc1 shape, desc2 shape", key1.shape, key2.shape, desc1.shape, desc2.shape)
        matched_keys1, matched_keys2 = self._match_descriptors(key1, desc1, key2, desc2)
        print("matched_keys1 shape, matched_keys2 shape", matched_keys1.shape, matched_keys2.shape)
        E_mat, inlier_matched_keys1, inlier_matched_keys2 = self._find_essential_matrix(matched_keys1, matched_keys2)
        print("E_mat shape, inlier_matched_keys1 shape, inlier_matched_keys2 shape", E_mat.shape, inlier_matched_keys1.shape, inlier_matched_keys2.shape)
        Rot_mat, translat = self._decompose_E(E_mat, inlier_matched_keys1, inlier_matched_keys2)
        print("Rot_mat shape, translat shape", Rot_mat.shape, translat.shape)
        landmarks = self._triangulate(Rot_mat, translat, inlier_matched_keys1, inlier_matched_keys2)
        print("landmarks shape", landmarks.shape)
        #TODO: Visualize function 

        S_1 = {
            # Structure in the second frame probably TODO: Clarify this
            'P': inlier_matched_keys2,       # shape (2, N)
            'X': landmarks,        # shape (3, N) or (4, N) in homogeneous
    
            # C_1 = candidate keypoints to be triangulated
            'C': np.empty((2, 0)),      # empty at the beginning
            'F': np.empty((2, 0)),      # first obs of each candidate
            'T': []                     # store the pose at first obs
        }
        T_WC_1 = {
            'R': Rot_mat,
            't': translat
        }

        return S_1, T_WC_1

    def run(self):
        """
        Example of a continuous loop over multiple frames. 
        frames = [I1, I2, I3, ...]
        """
        # 1) Initialize with first two frames
        S_1, T_WC_1 = self.initialize(
            # frames[0], frames[1],
        )
        self.global_poses.append(T_WC_1)

        # 2) For each subsequent frame
        S_prev = S_1
        T_prev = T_WC_1
        print("----------------------------------------INIT----------------------------------------")
        print("S: ", S_prev)
        print("T: ", T_prev)
        print('----------------------------------------INIT END----------------------------------------')
        print("Starting continuous operation...")
        for frame in self._get_next_frames():
            S_i, T_WC_i = self._process_frame(frame, S_prev, T_prev)
            self.global_poses.append(T_WC_i)
            S_prev = S_i
            T_prev = T_WC_i
            print("----------------------------------------CONT----------------------------------------")
            print("S: ", S_prev)
            print("T: ", T_prev)
            print('----------------------------------------CONT----------------------------------------')
        
        return self.global_poses

    def _detect_and_compute(self):
        """
        Detect keypoints and compute descriptors based on self.descriptor_name.
        """
        print(f"Descriptor: {self.descriptor_name}")
        
        if self.descriptor_name == 'harris':
            key1, desc1, key2, desc2, = self._detect_harris()
        elif self.descriptor_name == 'shi_tomasi':
            key1, desc1, key2, desc2 = self._detect_shi_tomasi()
        elif self.descriptor_name == 'sift':
            key1, desc1, key2, desc2 = self._detect_sift()
        else:
            raise ValueError("Invalid descriptor type")
        
        return key1, desc1, key2, desc2

    def _detect_harris(self):
        print("Harris")
        # For speed, read parameters only once into local variables
        harris_cfg = self.config['HARRIS']
        descriptors1, keypoints1 = get_descriptors_harris(
            self.img1,
            harris_cfg['corner_patch_size'], 
            harris_cfg['kappa'], 
            harris_cfg['num_keypoints'], 
            harris_cfg['nonmaximum_supression_radius'], 
            harris_cfg['descriptor_radius']
        )

        descriptors2, keypoints2 = get_descriptors_harris(
            self.img2,
            harris_cfg['corner_patch_size'], 
            harris_cfg['kappa'],
            harris_cfg['num_keypoints'], 
            harris_cfg['nonmaximum_supression_radius'],
            harris_cfg['descriptor_radius']
        )
        return keypoints1, descriptors1, keypoints2, descriptors2

    def _detect_shi_tomasi(self):
        st_cfg = self.config['SHI_TOMASI']
        descriptors1, keypoints1 = get_descriptors_st(
            self.img1,
            st_cfg['corner_patch_size'], 
            st_cfg['num_keypoints'],
            st_cfg['nonmaximum_supression_radius'], 
            st_cfg['descriptor_radius']
        )
        descriptors2, keypoints2 = get_descriptors_st(
            self.img2,
            st_cfg['corner_patch_size'], 
            st_cfg['num_keypoints'],
            st_cfg['nonmaximum_supression_radius'], 
            st_cfg['descriptor_radius']
        )
        return keypoints1, descriptors1, keypoints2, descriptors2

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

        keypoints1 = np.array([keypoints1_np[1], keypoints1_np[0]])  # swapped to match other things, not great TODO: fix
        keypoints2 = np.array([keypoints2_np[1], keypoints2_np[0]])
        descriptors1 = descriptors1_t
        descriptors2 = descriptors2_t

        return keypoints1, descriptors1, keypoints2, descriptors2
         
    def _match_descriptors(self, keys1, desc1, keys2, desc2):
        """
        Use matchDescriptors() from local library.
        Then extract matched keypoints in homogeneous coordinates.
        """
        if self.descriptor_name in ['harris', 'shi_tomasi']:
            match_lambda = self.config[self.descriptor_name.upper()]['match_lambda'] #TODO: not clean add one case for ST or harris independently
        else:
            match_lambda = self.config['SIFT']['match_lambda']

        # matches is a 1D array with length = # of query descriptors
        matches = matchDescriptors(
            desc2,  # Query
            desc1,  # Database TODO: This is swapped?
            match_lambda # Should be global?
        )
        print(f"Number of matches: {np.sum(matches != -1)}")

        query_indices = np.nonzero(matches >= 0)[0]
        match_indices = matches[query_indices]
        # print("query_indices shape:", query_indices.shape)
        # print("match_indices shape:", match_indices.shape)

        matched_keypoints1 = keys1[:, match_indices]
        matched_keypoints2 = keys2[:, query_indices]

        # Convert to homogeneous coords
        matched_keypoints1 = np.r_[matched_keypoints1, np.ones((1, matched_keypoints1.shape[1]))]
        matched_keypoints2 = np.r_[matched_keypoints2, np.ones((1, matched_keypoints2.shape[1]))]

        # Switch the coordinates if truly need them swapped again TODO: This is not great yet
        matched_keypoints1 = np.array([matched_keypoints1[1], matched_keypoints1[0], matched_keypoints1[2]])
        matched_keypoints2 = np.array([matched_keypoints2[1], matched_keypoints2[0], matched_keypoints2[2]])
        print("matched_keypoints2:\n", matched_keypoints2)
        return matched_keypoints1, matched_keypoints2

    def _find_essential_matrix(self, matched_keys1, matched_keys2):
        """
        Find the essential matrix using OpenCV's RANSAC-based findEssentialMat.
        """

        print("matched_keypoints1:", matched_keys1.shape)
        print("matched_keypoints2:", matched_keys2.shape)

        E_mat, mask = cv2.findEssentialMat(
            matched_keys1[:2].T,  # shape => Nx2
            matched_keys2[:2].T,
            self.K, 
            method=cv2.RANSAC, 
            prob=self.config['RANSAC']['prob'], 
            threshold=self.config['RANSAC']['threshold']
        )
        mask = mask.ravel() == 1 if mask is not None else None
        print("Mask shape:", mask.shape if mask is not None else "No mask")

        # Filter matched keypoints by inlier mask
        if mask is not None:
            matched_keys1 = matched_keys1[:, mask]
            matched_keys2 = matched_keys2[:, mask]
        else:
            print("No mask available.")

        return E_mat, matched_keys1, matched_keys2 

    def _decompose_E(self, E_mat, inlier_matched_keys1, inlier_matched_keys2):
        """
        Decompose the essential matrix into R, t.
        Disambiguate among the four possible solutions.
        """
        Rots, u3 = decomposeEssentialMatrix(E_mat)
        Rot_mat, translat = disambiguateRelativePose(
            Rots, u3, 
            inlier_matched_keys1, 
            inlier_matched_keys2, 
            self.K, self.K
        )
        print("Rotation matrix: ", Rot_mat)
        print("Translation vector: ", translat)
        return Rot_mat, translat
        
    def _triangulate(self, Rot_mat, translat, inlier_matched_keys1, inlier_matched_keys2):
        """
        Triangulate matched points into 3D using linearTriangulation.
        """
        M1 = self.K @ np.eye(3, 4)
        M2 = self.K @ np.c_[Rot_mat, translat]
        landmarks = linearTriangulation(
            inlier_matched_keys1,
            inlier_matched_keys2, 
            M1, 
            M2
        )
        return landmarks

    # Main part of the continuous operation
    # TODO: This is missing the logic for adding new landmarks.
    def _process_frame(self, frame, S_prev, T_prev):
        """
        Process each new frame for continuous operation.
        """
        S_new = S_prev.copy()
        T_new = T_prev.copy()
        curr_gray = frame  

        prev_keypoints = S_prev['P'][:2, :].T.reshape(-1, 1, 2).astype(np.float32)
        
        print(f"Tracking {len(prev_keypoints)} keypoints...")  # Debugging
        
        # Track keypoints
        valid_prev_keypoints, valid_curr_keypoints, valid_landmarks = track_keypoints(
            self.img2, curr_gray, prev_keypoints, S_prev['X'] # TODO: Correct here? Was self.P before
        )
        valid_prev_keypoints = valid_prev_keypoints.reshape(-1, 2)
        valid_curr_keypoints = valid_curr_keypoints.reshape(-1, 2)

        # Use PnP with ransac
        landmarks = valid_landmarks
        landmarks = landmarks[:, :3]  # Take only the first 3 columns (x, y, z)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            landmarks,  # 3D
            valid_curr_keypoints,  # 2D
            self.K,
            None,
            iterationsCount=self.config['PNPRANSAC']['iterations'],
            reprojectionError=self.config['PNPRANSAC']['reprojection_error'],
            confidence=self.config['PNPRANSAC']['prob'],
            flags=cv2.SOLVEPNP_EPNP
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            self.R = R
            self.t = t
            T_new['R'] = R
            T_new['t'] = t
            # TODO: Output of this is not good. At the start i think it is acceptable, then it gets horrible at some point.
            # Might be because there are almost no landmarks left anymore at some point
            print("Pose:")
            print(R)
            print(t)
            # Filter valid keypoints using the inliers from solvePnPRansac
            valid_curr_keypoints = valid_curr_keypoints[inliers.ravel()]
            valid_prev_keypoints = valid_prev_keypoints[inliers.ravel()]
            valid_landmarks = valid_landmarks[inliers.ravel()]

            # Convert into homogeneous coordinates
            valid_prev_keypoints = np.r_[valid_prev_keypoints.T, np.ones((1, valid_prev_keypoints.shape[0]))]
            valid_curr_keypoints = np.r_[valid_curr_keypoints.T, np.ones((1, valid_curr_keypoints.shape[0]))]

            # Update keypoints
            S_new['P'] = valid_curr_keypoints
            S_new['X'] = valid_landmarks.T
            return S_new, T_new

            # self.matched_keypoints1 = valid_prev_keypoints
            # self.matched_keypoints2 = valid_curr_keypoints
            # self.P = valid_landmarks.T
        else:
            print("Pose konnte nicht berechnet werden.")

    def _get_next_frames(self):
        """
        Generator to yield frames from the dataset for continuous operation.
        """
        dataset_path = os.path.join(self.data_rootdir, self.dataset_curr, 'images')
        image_files = sorted([
            os.path.join(dataset_path, f) 
            for f in os.listdir(dataset_path) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        init_img_2_path = self.config["DATA"]["init_img_2"]
        init_img_2_filename = os.path.basename(init_img_2_path)
        image_filenames = [os.path.basename(f) for f in image_files]

        start_index = image_filenames.index(init_img_2_filename)
        # Skip the firstframes (used for initialization)
        for img_path in image_files[start_index:]:
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"Failed to load {img_path}. Skipping...")
                continue
            yield frame

if __name__ == "__main__":
    pipeline = VisualOdometryPipeline(config_path='Code/config.yaml')
    pipeline.run()
