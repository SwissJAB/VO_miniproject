import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
import os

from descriptor_utils import get_descriptors_harris, get_descriptors_st, get_descriptors_harris_cv2, get_descriptors_st_cv2
from get_descriptors.match_descriptors import matchDescriptors
from get_descriptors.plot_matches import plotMatches

from two_view_geometry.estimate_essential_matrix import estimateEssentialMatrix
from two_view_geometry.decompose_essential_matrix import decomposeEssentialMatrix
from two_view_geometry.disambiguate_relative_pose import disambiguateRelativePose
from two_view_geometry.linear_triangulation import linearTriangulation
from two_view_geometry.draw_camera import drawCamera

from feature_tracking.klt_tracking import track_keypoints
from visualizer import VisualOdometryVisualizer


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
        self.visualizer = VisualOdometryVisualizer()


    def initialize(self):
        #delete previous folders
        if self.config["PLOTS"]["save"]:
            save_path = self.config["PLOTS"]["save_path"]
            if os.path.exists(save_path):
                for file in os.listdir(save_path):
                    file_path = os.path.join(save_path, file)
                    try:
                        if os.path.isfile(file_path) and (file.startswith('pose') or file.startswith('inliers')) and self.descriptor_name in file:
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)
            else:
                os.makedirs(save_path)

        key1, desc1, key2, desc2 = self._detect_and_compute_init()
        print("key1 shape, key2 shape, desc1 shape, desc2 shape", key1.shape, key2.shape, desc1.shape, desc2.shape)
        matched_keys1, matched_keys2 = self._match_descriptors(key1, desc1, key2, desc2)
        print("matched_keys1 shape, matched_keys2 shape", matched_keys1.shape, matched_keys2.shape)
        E_mat, inlier_matched_keys1, inlier_matched_keys2 = self._find_essential_matrix(matched_keys1, matched_keys2)
        print("E_mat shape, inlier_matched_keys1 shape, inlier_matched_keys2 shape", E_mat.shape, inlier_matched_keys1.shape, inlier_matched_keys2.shape)
        Rot_mat, translat = self._decompose_E(E_mat, inlier_matched_keys1, inlier_matched_keys2)
        print("Rot_mat shape, translat shape", Rot_mat.shape, translat.shape)
        landmarks = self._triangulate(Rot_mat, translat, inlier_matched_keys1, inlier_matched_keys2)
        print("landmarks shape", landmarks.shape)

        S_1 = {
            # Structure in the second frame probably TODO: Clarify this
            'P': inlier_matched_keys2,       # shape (2, N)
            'X': landmarks,        # shape (3, N) or (4, N) in homogeneous
    
            # C_1 = candidate keypoints to be triangulated
            'C': np.empty((2, 0)),      # empty at the beginning
            'F': np.empty((2, 0)),      # first obs of each candidate
            'T': np.empty((1,0))                     # store the pose at first obs
        }
        T_WC_1 = {
            'R': Rot_mat,
            't': translat,
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
        prev_frame = self.img2
        for frame in self._get_next_frames():
            S_i, T_WC_i = self._process_frame(frame, prev_frame, S_prev, T_prev)
            self.global_poses.append(T_WC_i)
            print("X:", S_i['X'].shape)
            print("P:", S_i['P'].shape)
            print("T:", S_i['T'].shape)
            print("C:", S_i['C'].shape)
            print("F:", S_i['F'].shape)

            S_prev = S_i
            T_prev = T_WC_i
            prev_frame = frame
            self.visualizer.update_visualizations(S_i['X'], T_WC_i['t'], frame, S_i['P'])
            print("----------------------------------------CONT----------------------------------------")
            # print("S: ", S_prev)
            print("T: ", T_prev)
            if self.config["PLOTS"]["save"]:
                # save pose in txt file of current descriptor and dataset
                pose_path = os.path.join(self.config["PLOTS"]["save_path"], f"pose_{self.descriptor_name}_{self.dataset_curr}.txt")
                with open(pose_path, 'a') as f:
                    f.write(" ".join(map(str, T_prev['R'][0,:])) + " " + str(T_prev['t'][0])+ " ")
                    f.write(" ".join(map(str, T_prev['R'][1,:])) + " " + str(T_prev['t'][1])+ " ")
                    f.write(" ".join(map(str, T_prev['R'][2,:])) + " " + str(T_prev['t'][2]) + "\n")
            print('----------------------------------------CONT----------------------------------------')
        
        self.visualizer.close()
        return self.global_poses

    def _detect_and_compute_init(self):
        """
        Detect keypoints and compute descriptors based on self.descriptor_name.
        """
        print(f"Descriptor: {self.descriptor_name}")
        
        if self.descriptor_name == 'harris':
            key1, desc1 = self._detect_harris(self.img1) 
            key2, desc2 = self._detect_harris(self.img2)
        elif self.descriptor_name == 'shi_tomasi':
            key1, desc1 = self._detect_shi_tomasi(self.img1) 
            key2, desc2 = self._detect_shi_tomasi(self.img2)
        elif self.descriptor_name == 'sift':
            key1, desc1 = self._detect_sift(self.img1)
            key2, desc2 = self._detect_sift(self.img2)
        else:
            raise ValueError("Invalid descriptor type")
        
        return key1, desc1, key2, desc2

    def _detect_harris(self, img):
        if self.config["PLOTS"]["save"]:
            start_time = time.time()
        print("Harris")
        # For speed, read parameters only once into local variables
        harris_cfg = self.config['HARRIS']
        if harris_cfg['cv2']:
            descriptors, keypoints = get_descriptors_harris_cv2(
                img,
                harris_cfg['corner_patch_size'], 
                harris_cfg['kappa'], 
                harris_cfg['num_keypoints'], 
                harris_cfg['nonmaximum_supression_radius'], 
                harris_cfg['descriptor_radius']
            )
        else:
            descriptors, keypoints = get_descriptors_harris(
                img,
                harris_cfg['corner_patch_size'], 
                harris_cfg['kappa'], 
                harris_cfg['num_keypoints'], 
                harris_cfg['nonmaximum_supression_radius'], 
                harris_cfg['descriptor_radius']
            )

        if self.config["PLOTS"]["save"]:
            end_time = time.time()
            save_path = self.config["PLOTS"]["save_path"]
            with open(os.path.join(save_path, f"time_{self.descriptor_name}.txt"), 'a') as f:
                f.write(f"{end_time - start_time}\n")
        return keypoints, descriptors

    def _detect_shi_tomasi(self, img):
        if self.config["PLOTS"]["save"]:
            start_time = time.time()
        st_cfg = self.config['SHI_TOMASI']

        if st_cfg['cv2']:
            descriptors, keypoints = get_descriptors_st_cv2(
                img,
                st_cfg['corner_patch_size'], 
                st_cfg['num_keypoints'],
                st_cfg['nonmaximum_supression_radius'], 
                st_cfg['descriptor_radius']
            )
        else:
            descriptors, keypoints = get_descriptors_st(
                img,
                st_cfg['corner_patch_size'], 
                st_cfg['num_keypoints'],
                st_cfg['nonmaximum_supression_radius'], 
                st_cfg['descriptor_radius']
            )
        if self.config["PLOTS"]["save"]:
            end_time = time.time()
            save_path = self.config["PLOTS"]["save_path"]
            with open(os.path.join(save_path, f"time_{self.descriptor_name}.txt"), 'a') as f:
                f.write(f"{end_time - start_time}\n")
    
        return keypoints, descriptors

    def _detect_sift(self, img):
        if self.config["PLOTS"]["save"]:
            start_time = time.time()
        sift_cfg = self.config['SIFT']
        sift = cv2.SIFT_create(
            nfeatures=sift_cfg['nfeatures'],
            contrastThreshold=sift_cfg['contrast_threshold'],
            sigma=sift_cfg['sigma'],
            nOctaveLayers=sift_cfg['n_otave_layers']
        )
        kp1, desc1 = sift.detectAndCompute(img, None)

        # Convert to np arrays
        keypoints = np.array([kp.pt for kp in kp1]).T  # shape (2, N1)
        descriptors = desc1.T                       # shape (128, N1)

        if self.config["PLOTS"]["save"]:
            end_time = time.time()
            save_path = self.config["PLOTS"]["save_path"]
            with open(os.path.join(save_path, f"time_{self.descriptor_name}.txt"), 'a') as f:
                f.write(f"{end_time - start_time}\n")

        return keypoints, descriptors
         
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

        matched_keypoints1 = keys1[:, match_indices]
        matched_keypoints2 = keys2[:, query_indices]

        # Convert to homogeneous coords
        matched_keypoints1 = np.r_[matched_keypoints1, np.ones((1, matched_keypoints1.shape[1]))]
        matched_keypoints2 = np.r_[matched_keypoints2, np.ones((1, matched_keypoints2.shape[1]))]

        matched_keypoints1 = np.array([matched_keypoints1[0], matched_keypoints1[1], matched_keypoints1[2]])
        matched_keypoints2 = np.array([matched_keypoints2[0], matched_keypoints2[1], matched_keypoints2[2]])
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
    def _process_frame(self, frame, prev_frame, S_prev, T_prev):
        """
        Process each new frame for continuous operation.
        """
        S_new = S_prev.copy() # TODO: Make sure we put new stuff in here
        T_new = T_prev.copy()
        curr_gray = frame  
    
    # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(21, 21),
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Calculate optical flow
        if S_prev['C'].shape[1] > 0:
            S_prev['C'] = S_prev['C'][:2, :].T.reshape(-1, 1, 2).astype(np.float32)
            tracked_candidate_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_gray, S_prev['C'], None, **lk_params)
            tracked_candidate_keypoints = tracked_candidate_keypoints.reshape(-1, 2)
            S_new['C'] = tracked_candidate_keypoints[status.flatten() == 1]
            S_new['C'] = S_new['C'].T
            print("T shape", S_prev['T'].shape)
            print("F shape", S_prev['F'].shape)
            S_new['F'] = S_prev['F'][:, status.flatten() == 1]
            S_new['T'] = S_prev['T'][:, status.flatten() == 1]

            # Find new candidate keypoints
        if self.descriptor_name == 'harris':
            candidate_keypoints, _ = self._detect_harris(curr_gray) 
        elif self.descriptor_name == 'shi_tomasi':
            candidate_keypoints, _ = self._detect_shi_tomasi(curr_gray) 
        elif self.descriptor_name == 'sift':
            candidate_keypoints, _ = self._detect_sift(curr_gray)
        
        # Add the new candidate keypoints to the candidate keypoints
        print("Candidate keypoints shape:", candidate_keypoints.shape)
        candidate_keypoints = self._remove_duplicates(candidate_keypoints, S_new['P'])
        candidate_keypoints = self._remove_duplicates(candidate_keypoints, S_new['C'])
        print("Candidate keypoints shape after removing duplicates:", candidate_keypoints.shape)
        S_new['C'] = np.c_[S_new['C'], candidate_keypoints]
        # Add the new candidate keypoints to the first observation
        S_new['F'] = np.c_[S_new['F'], candidate_keypoints]

        # can_size = candidate_keypoints.shape[1]
        # if can_size> 0:
        #     T_new_repeated = np.empty((1, can_size), dtype=object)
        #     for j in range(can_size):
        #         # .copy() just to be safe that each column is a separate dict
        #         T_new_repeated[0, j] = T_new.copy()
        #     S_new['T'] = np.concatenate([S_new['T'], T_new_repeated], axis=1)

        prev_keypoints = S_prev['P'][:2, :].T.reshape(-1, 1, 2).astype(np.float32)
        
        print(f"Tracking {len(prev_keypoints)} keypoints...")  # Debugging
        
        # Track keypoints
        valid_prev_keypoints, valid_curr_keypoints, valid_landmarks = track_keypoints(
            self.img2, curr_gray, prev_keypoints, S_prev['X']
        )
        valid_prev_keypoints = valid_prev_keypoints.reshape(-1, 2)
        valid_curr_keypoints = valid_curr_keypoints.reshape(-1, 2)

        # Use PnP with ransac
        landmarks = valid_landmarks
        landmarks = landmarks[:, :3]  # Take only the first 3 columns (x, y, z)
        print("Landmarks shape:", landmarks.shape)
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
        print("Inliers shape:", inliers.shape)
        if self.config["PLOTS"]["save"]:
            save_path = self.config["PLOTS"]["save_path"]
            with open(os.path.join(save_path, f"inliers_{self.descriptor_name}.txt"), 'a') as f:
                f.write(f"{landmarks.shape[0]} {inliers.shape[0]}\n")
        if success:
            R_mat, _ = cv2.Rodrigues(rvec)
            t_flat = tvec.flatten()
            T_new['R'] = R_mat
            T_new['t'] = t_flat
            print("Pose:")
            print(R_mat)
            print(t_flat)


            T_new_repeated = np.tile(T_new, (len(candidate_keypoints[1]), 1))
            print("T_new_repeated shape:", T_new_repeated.shape)
            S_new['T'] = np.c_[S_new['T'], T_new_repeated.T]

            ### Logic for adding new landmarks
            new_3d_points = []
            new_3d_points_2d = []

            if S_new['T'].size > 0:
                indices_to_remove = []
                for i in range(S_new['C'].shape[1]):
                    c = S_new['C'][:, i].reshape(2, 1)  # shape (2,1)
                    f = S_new['F'][:, i].reshape(2, 1)  # shape (2,1)

                    t = S_new['T'][:, i]
                    angle_c = self._compute_angle(f, c, t[0].copy(), T_new.copy(), self.K)
                    if angle_c > self.baseline_angle_thresh:
                        M1 = self.K @ np.c_[t[0]['R'], t[0]['t']]
                        M2 = self.K @ np.c_[T_new['R'], T_new['t']]
                        homogeneous_f = np.vstack((f, [1]))
                        homogeneous_c = np.vstack((c, [1]))
                        new_3d_point = linearTriangulation(homogeneous_f, homogeneous_c, M1, M2) # TODO: Should be the correct order right?
                        new_3d_points.append(new_3d_point)
                        new_3d_points_2d.append(c)
                        indices_to_remove.append(i)


            if len(new_3d_points) > 0:
                print(f"if 1: Adding {len(new_3d_points)} new landmarks.")
                print("New 3D points shape before stack:", new_3d_points[0].shape)
                print("New 3D points 2D shape before stack:", new_3d_points_2d[0].shape)
                new_3d_points = np.column_stack(new_3d_points)
                new_3d_points_2d = np.column_stack(new_3d_points_2d)
                new_3d_points_2d_h = np.r_[new_3d_points_2d, np.ones((1, new_3d_points_2d.shape[1]))]
                print("New 3D points shape:", new_3d_points.shape)
                print("New 3D points 2D shape:", new_3d_points_2d_h.shape)

                # Append to the existing landmarks
                #S_new['X'] = np.c_[S_new['X'], new_3d_points]

                # Also keep track of the 2D “feature-plane” location in P if you want
                #S_new['P'] = np.c_[S_new['P'], np.r_[new_3d_points_2d, np.ones((1, new_3d_points_2d.shape[1]))]]

                # Remove these candidates from 'C' and 'F'
                mask = np.ones(S_new['C'].shape[1], dtype=bool)
                mask[indices_to_remove] = False
                S_new['C'] = S_new['C'][:, mask]
                S_new['F'] = S_new['F'][:, mask]
                S_new['T'] = S_new['T'][:, mask]


            ### Logic for PnP

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

            if len(new_3d_points) > 0:
                print(f"If 2: Adding {new_3d_points.shape[1]} new landmarks.")
                print("New 3D points shape:", new_3d_points.shape)
                print("New 3D points 2D shape:", new_3d_points_2d_h.shape)
                S_new['P'] = np.c_[S_new['P'], new_3d_points_2d_h]
                S_new['X'] = np.c_[S_new['X'], new_3d_points]

            return S_new, T_new
        else:
            print("Pose was not calculated.")

    # TODO: This might be a suboptimal way to remove duplicates
    def _remove_duplicates(self, new_kps, existing_kps):
        """
        new_kps:      shape (2, N_new)
        existing_kps: shape (2, N_exist)
        returns:      shape (2, M) subset of new_kps that are not duplicates
        """
        if existing_kps.shape[1] == 0:
            return new_kps  # If no existing kps, all are unique

        unique_kps = []
        for i in range(new_kps.shape[1]):
            kp = new_kps[:, i]
            # Euclidean distance TODO: Maybe suboptimal, not sure rn
            distances = np.linalg.norm(existing_kps[:2, :] - kp.reshape(2, 1), axis=0)
            if np.min(distances) > self.config['CONT_VO']['kp_dist_thresh']:
                unique_kps.append(kp)

        if len(unique_kps) == 0:
            return np.empty((2, 0))  # No unique kps
        else:
            return np.column_stack(unique_kps)




    def _compute_angle(self, f, c, T_i, T_j, K):
        """
        f, c:   (2,1) pixel coordinates of the same 3D point in first/new frames
        T_i:    dict with {'R': R_i, 't': t_i} for the first camera pose
        T_j:    dict with {'R': R_j, 't': t_j} for the new camera pose
        K:      (3,3) intrinsic matrix

        returns: float, angle [radians] between the two bearing vectors in world frame
        """
        f_h = np.vstack((f, [1]))  # shape (3,1)
        c_h = np.vstack((c, [1]))  # shape (3,1)

        # 2) Back-project to normalized camera rays
        f_cam = np.linalg.inv(K) @ f_h
        c_cam = np.linalg.inv(K) @ c_h

        R_i = T_i['R']  # shape (3,3)
        R_j = T_j['R']  # shape (3,3)
        v_i_world = R_i @ f_cam  # shape (3,1)
        v_j_world = R_j @ c_cam  # shape (3,1)

        dotp = np.dot(v_i_world.ravel(), v_j_world.ravel())
        denom = np.linalg.norm(v_i_world) * np.linalg.norm(v_j_world)
        # Numerical guard
        cosine = np.clip(dotp / (denom + 1e-15), -1.0, 1.0)
        angle = np.arccos(cosine)
        #print("Angle:", angle, "Sucess:", angle > self.baseline_angle_thresh)
        return angle



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
