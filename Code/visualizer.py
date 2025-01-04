import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class VisualOdometryVisualizer2:
    def __init__(self):
        # Initialize the figure
        self.fig = plt.figure(figsize=(12, 6))

        # Set up the axes for plotting
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # 3D plot
        self.ax2 = self.fig.add_subplot(122)  # 2D image plot

        # Define fixed axis limits for the 3D plot
        self.ax1.set_xlim([-5, 10])  # Set X-axis limit
        self.ax1.set_ylim([-5, 10])  # Set Y-axis limit
        self.ax1.set_zlim([-5, 10])  # Set Z-axis limit

        self.camera_trajectory = []
        self.view_set = False

    def update_visualizations(self, landmarks_3d, camera_pose, image, keypoints):
        """
        Updates the figure with 3D landmarks and 2D keypoints.
        
        Parameters:
        landmarks_3d: 3xN array of 3D landmarks
        camera_pose: camera pose [x, y, z]
        image: 2D array (image being processed)
        keypoints: 2xN array of keypoints (x, y)
        """
        camera_pose = -camera_pose
        self.camera_trajectory.append(camera_pose)
        # Clear previous plots
        self.ax1.cla()  # Clear the 3D plot
        self.ax2.cla()  # Clear the 2D image plot

        traj = np.array(self.camera_trajectory)
        self.ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='r', label='Camera Trajectory')

        # 3D Plot: Landmarks and Camera Pose
        self.ax1.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], c='b', label='Landmarks')
        self.ax1.scatter(camera_pose[0], camera_pose[1], camera_pose[2], c='r', label='Camera Pose')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()

        # Set the axis limits to prevent rescaling
        self.ax1.set_xlim([-5, 10])  # Keep fixed limits for X-axis
        self.ax1.set_ylim([-5, 10])  # Keep fixed limits for Y-axis
        self.ax1.set_zlim([-5, 10])  # Keep fixed limits for Z-axis

        if not self.view_set and len(traj) > 0:
            self.ax1.view_init(elev=90, azim=180)  # Adjust these values as needed
            self.view_set = True

        # 2D Plot: Image and Keypoints
        self.ax2.imshow(image, cmap='gray')  # Image in grayscale
        self.ax2.scatter(keypoints[:, 0], keypoints[:, 1], c='g', marker='o', label='Keypoints')
        self.ax2.set_title('Processed Image with Keypoints')

        # Show the plot
        plt.draw()
        plt.pause(0.001)  # Pause for a moment to update the plot


class VisualOdometryVisualizer:
    def __init__(self):
        # Initialize the figure
        self.fig = plt.figure(figsize=(12, 6))

        # Set up the axes for plotting
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # 3D plot
        self.ax2 = self.fig.add_subplot(122)  # 2D image plot

        # Define fixed axis limits for the 3D plot
        self.ax1.set_xlim([-10, 15])  # Set X-axis limit
        self.ax1.set_ylim([-10, 15])  # Set Y-axis limit
        self.ax1.set_zlim([-10, 15])  # Set Z-axis limit

        self.camera_trajectory = []
        self.previous_landmarks = None

    def update_visualizations(self, landmarks_3d, camera_pose, image, keypoints):
        """
        Updates the figure with 3D landmarks and 2D keypoints.
        
        Parameters:
        landmarks_3d: 3xN array of 3D landmarks
        camera_pose: camera pose [x, y, z]
        image: 2D array (image being processed)
        keypoints: 2xN array of keypoints (x, y)
        """
        camera_pose = -camera_pose
        self.camera_trajectory.append(camera_pose)
        
        # Clear previous plots
        self.ax1.cla()  # Clear the 3D plot
        self.ax2.cla()  # Clear the 2D image plot

        traj = np.array(self.camera_trajectory)
        
        # Plot camera trajectory
        self.ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='r', label='Camera Trajectory')

        # Plot previously seen landmarks in green
        if self.previous_landmarks is not None:
            self.ax1.scatter(
                self.previous_landmarks[:, 0], 
                self.previous_landmarks[:, 1], 
                self.previous_landmarks[:, 2], 
                c='g', label='Previous Landmarks'
            )

        # Plot current landmarks in blue
        self.ax1.scatter(
            landmarks_3d[:, 0], 
            landmarks_3d[:, 1], 
            landmarks_3d[:, 2], 
            c='b', label='Current Landmarks'
        )

        # Update previous landmarks
        self.previous_landmarks = landmarks_3d

        # Plot current camera pose
        self.ax1.scatter(camera_pose[0], camera_pose[1], camera_pose[2], c='r', s=10, label='Camera Pose')

        # Set axis labels
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()

        # Dynamically adjust the view to follow the camera
        self.ax1.view_init(elev=20, azim=90)
        self.ax1.set_xlim([camera_pose[0] - 5, camera_pose[0] + 5])
        self.ax1.set_ylim([camera_pose[1] - 5, camera_pose[1] + 5])
        self.ax1.set_zlim([camera_pose[2] - 5, camera_pose[2] + 5])

        # 2D Plot: Image and Keypoints
        self.ax2.imshow(image, cmap='gray')  # Image in grayscale
        self.ax2.scatter(keypoints[:, 0], keypoints[:, 1], c='g', marker='o', label='Keypoints')
        self.ax2.set_title('Processed Image with Keypoints')

        # Show the plot
        plt.draw()
        plt.pause(0.001)  # Pause for a moment to update the plot