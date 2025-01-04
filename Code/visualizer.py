import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

class VisualOdometryVisualizer:
    def __init__(self):
        # Initialize the figure
        self.fig = plt.figure(figsize=(15, 12))  # Adjusted size for new layout

        # Set up the axes for plotting
        self.ax1 = self.fig.add_subplot(231, projection='3d')  # 3D plot with fixed limits
        self.ax2 = self.fig.add_subplot(2, 3, (2, 3))
        self.ax3 = self.fig.add_subplot(234, projection='3d')  # 3D plot without clearing
        self.ax4 = self.fig.add_subplot(235)  # 2D plot for number of landmarks
        self.ax5 = self.fig.add_subplot(236)  # 2D plot for FPS

        # Define fixed axis limits for the 3D plot
        self.ax1.set_xlim([-10, 15])  # Set X-axis limit
        self.ax1.set_ylim([-10, 15])  # Set Y-axis limit
        self.ax1.set_zlim([-10, 15])  # Set Z-axis limit

        self.camera_trajectory = []
        self.landmark_counts = []
        self.keypoints_number = []
        self.fps_values = []
        self.last_update_time = time.time()
        

    def update_visualizations(self, landmarks_3d, camera_pose, image, keypoints):
        """
        Updates the figure with 3D landmarks and 2D keypoints.
        
        Parameters:
        landmarks_3d: 3xN array of 3D landmarks
        camera_pose: camera pose [x, y, z]
        image: 2D array (image being processed)
        keypoints: 2xN array of keypoints (x, y)
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        fps = 1.0 / time_elapsed if time_elapsed > 0 else 0
        self.fps_values.append(fps)
        self.last_update_time = current_time

        self.camera_trajectory.append(camera_pose)
            # Keep only the last 20 values

        if len(self.fps_values) > 20:
            self.fps_values = self.fps_values[-20:]

        self.landmark_counts.append(landmarks_3d.shape[0])
        if len(self.landmark_counts) > 20:
            self.landmark_counts = self.landmark_counts[-20:]
            
        self.keypoints_number.append(keypoints.shape[0])
        if len(self.keypoints_number) > 20:
            self.keypoints_number = self.keypoints_number[-20:]

        self.last_update_time = current_time

        # if keypoints.shape[0] >1000, randomly take 1000 keypoints to plot
        if keypoints.shape[0] > 750 or landmarks_3d.shape[0] > 750:
            idx = np.random.choice(keypoints.shape[0], 750, replace=False)
            keypoints = keypoints[idx]
            landmarks_3d = landmarks_3d[idx]
        
        # Clear previous plots
        self.ax1.cla()  # Clear the 3D plot with fixed limits
        self.ax2.cla()  # Clear the 2D image plot
        self.ax4.cla()  # Clear the landmarks count plot
        self.ax5.cla()  # Clear the FPS plot

        traj = np.array(self.camera_trajectory)
        
        # Plot camera trajectory
        self.ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='r', label='Camera Trajectory')

        # Plot current landmarks in blue
        self.ax1.scatter(
            landmarks_3d[:, 0], 
            landmarks_3d[:, 1], 
            landmarks_3d[:, 2], 
            c='b', label='Current Landmarks'
        )

        # Plot current camera pose
        self.ax1.scatter(camera_pose[0], camera_pose[1], camera_pose[2], c='r', s=10, label='Camera Pose')

        # Set axis labels
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()
        self.ax1.set_title('3D Plot of Local Camera Trajectory and Landmarks')

        # Dynamically adjust the view to follow the camera
        self.ax1.view_init(elev=0, azim=270)
        self.ax1.set_xlim([camera_pose[0] - 15, camera_pose[0] + 15])
        self.ax1.set_ylim([camera_pose[1] - 15, camera_pose[1] + 15])
        self.ax1.set_zlim([camera_pose[2] - 15, camera_pose[2] + 15])

        # 2D Plot: Image and Keypoints
        self.ax2.imshow(image, cmap='gray')  # Image in grayscale
        self.ax2.scatter(keypoints[:, 0], keypoints[:, 1], c='g', marker='o', label='Keypoints', s=5)
        self.ax2.set_title('Processed Image with Keypoints (Capped at 750 for Efficient Plotting)')

        # 3D Plot without clearing
        self.ax3.scatter(camera_pose[0], camera_pose[1], camera_pose[2], c='r', s=5, label='Camera Pose')
        self.ax3.set_xlabel('X')
        self.ax3.set_ylabel('Y')
        self.ax3.set_zlabel('Z')
        self.ax3.view_init(elev=0, azim=270)

        # Dynamically adjust the view to follow the camera for ax3
        max_range = np.array([traj[:, 0].max() - traj[:, 0].min(), 
                        traj[:, 1].max() - traj[:, 1].min(), 
                        traj[:, 2].max() - traj[:, 2].min()]).max() / 2.0

        mid_x = (traj[:, 0].max() + traj[:, 0].min()) * 0.5
        mid_y = (traj[:, 1].max() + traj[:, 1].min()) * 0.5
        mid_z = (traj[:, 2].max() + traj[:, 2].min()) * 0.5

        self.ax3.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax3.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax3.set_zlim(mid_z - max_range, mid_z + max_range)
        self.ax3.set_title('Global Trajectory of Camera Pose')

        # Plot the last 20 keypoints over frames
        if len(self.keypoints_number) > 20:
            self.ax4.plot(range(len(self.keypoints_number) - 20, len(self.keypoints_number)), 
                  self.keypoints_number[-20:], c='b', label='Last 20 Keypoints')
        else:
            self.ax4.plot(range(len(self.keypoints_number)), 
                  self.keypoints_number, c='b', label='Last Keypoints')

        self.ax4.set_xlabel('Frame Number')
        self.ax4.set_ylabel('Number of Keypoints')
        self.ax4.set_title('Number of Keypoints per Frame')
        #self.ax4.legend()

        # Plot the last 20 FPS values
        if len(self.fps_values) > 20:
            self.ax5.plot(range(len(self.fps_values) - 20, len(self.fps_values)), 
                  self.fps_values[-20:], c='g', label='Last 20 FPS')
        else:
            self.ax5.plot(range(len(self.fps_values)), 
                  self.fps_values, c='g', label='FPS')

        self.ax5.set_xlabel('Frame Number')
        self.ax5.set_ylabel('FPS')
        self.ax5.set_title('FPS per Frame of the Last 20 Frames')
        #self.ax5.legend()

        # Show the plot
        plt.draw()
        plt.pause(0.0001)  # Pause for a moment to update the plot

