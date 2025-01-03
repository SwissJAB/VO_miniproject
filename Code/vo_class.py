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


    def initialize_state(self, keypoints, landmarks, R, t):
        """
        Initialize the state with keypoints, landmarks, and pose.
        
        Args:
            keypoints (np.ndarray): Initial keypoints (2, N)
            landmarks (np.ndarray): Initial 3D landmarks (3, N)
            R (np.ndarray): Initial rotation matrix (3, 3)
            t (np.ndarray): Initial translation vector (3, 1)
        """
        # Convert 2D keypoints to homogeneous coordinates (3, N)
        keypoints_homogeneous = np.r_[keypoints, np.ones((1, keypoints.shape[1]))]  # (3, N)
        self.matched_keypoints2 = keypoints_homogeneous

        self.P = landmarks
        self.R = R
        self.t = t

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


        self.initialize_state(
            self.matched_keypoints2[:2],
            self.P,
            self.R,
            self.t
        )

        # Continuous Operation Phase
        for frame in self._get_next_frames():
            # Show the next frame
            self._process_frame(frame)
            self.img2 = frame
            self._visualize()
        
        if self.config["PLOTS"]["save"]:
            # create a video from the images and then save that video in the same folder and delete the images

            # create a video from the images
            img_folder = os.path.join(self.config["PLOTS"]["save_path"], "images4video")
            video_path = os.path.join(self.config["PLOTS"]["save_path"], "video.mp4")
            
            # Check if there are any images in the folder
            if any(fname.endswith('.png') for fname in os.listdir(img_folder)):
                # print current working directory
                print("Current working directory:", os.getcwd())
                
                os.system(f"ffmpeg -r 1 -pattern_type glob -i '{img_folder}/*.png' -vcodec mpeg4 -y {video_path}")

                # delete the images
                for img_file in os.listdir(img_folder):
                    img_path = os.path.join(img_folder, img_file)
                    if os.path.isfile(img_path) and img_file.endswith('.png'):
                        os.remove(img_path)
            else:
                print(f"No images found in {img_folder}. Skipping video creation.")


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
        print("shi keypoints1 shape:", self.keypoints1.shape)
        print("shi keypoints2 shape:", self.keypoints2.shape)
        print("shi descriptors1 shape:", self.descriptors1.shape)
        print("shi descriptors2 shape:", self.descriptors2.shape)

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

        print("SIFT keypoints1 shape:", self.keypoints1.shape)
        print("SIFT keypoints2 shape:", self.keypoints2.shape)
        print("SIFT descriptors1 shape:", self.descriptors1.shape)
        print("SIFT descriptors2 shape:", self.descriptors2.shape)
         
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
            prob=self.config['RANSAC']['prob'], 
            threshold=self.config['RANSAC']['threshold']
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
        self.img1 = self.img2

        if self.config["PLOTS"]["save"]:
            save_dir = os.path.join(self.config["PLOTS"]["save_path"], "images4video")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, str(int(time.time()*1000)) + ".png"))
        
        if self.config["PLOTS"]["show"]:
            plt.show()

    # Main part of the continuous operation
    # TODO: This is missing the logic for adding new landmarks.
    def _process_frame(self, frame):
        """
        Process each new frame for continuous operation.
        """
        curr_gray = frame  

        prev_keypoints = self.matched_keypoints2[:2, :].T.reshape(-1, 1, 2).astype(np.float32)  # (N, 1, 2)
        
        print(f"Tracking {len(prev_keypoints)} keypoints...")  # Debugging
        
        # Track keypoints
        valid_prev_keypoints, valid_curr_keypoints, valid_landmarks = track_keypoints(
            self.img2, curr_gray, prev_keypoints, self.P
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
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=self.config['RANSAC']['prob'],
            flags=cv2.SOLVEPNP_EPNP
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            self.R = R
            self.t = t
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
            self.matched_keypoints1 = valid_prev_keypoints
            self.matched_keypoints2 = valid_curr_keypoints
            self.P = valid_landmarks.T
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



