import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class VisualOdometryVisualizer:
    def __init__(self):
        # Initialize the figure
        self.fig = plt.figure(figsize=(18, 6))

        # Set up the axes for plotting
        self.ax1 = self.fig.add_subplot(131, projection='3d')  # 3D plot with fixed limits
        self.ax2 = self.fig.add_subplot(132)  # 2D image plot
        self.ax3 = self.fig.add_subplot(133, projection='3d')  # 3D plot without clearing

        # Define fixed axis limits for the 3D plot
        self.ax1.set_xlim([-10, 15])  # Set X-axis limit
        self.ax1.set_ylim([-10, 15])  # Set Y-axis limit
        self.ax1.set_zlim([-10, 15])  # Set Z-axis limit

        self.camera_trajectory = []
        # self.previous_landmarks = None

    def update_visualizations(self, landmarks_3d, camera_pose, image, keypoints):
        """
        Updates the figure with 3D landmarks and 2D keypoints.
        
        Parameters:
        landmarks_3d: 3xN array of 3D landmarks
        camera_pose: camera pose [x, y, z]
        image: 2D array (image being processed)
        keypoints: 2xN array of keypoints (x, y)
        """
        self.camera_trajectory.append(camera_pose)
        
        # Clear previous plots
        self.ax1.cla()  # Clear the 3D plot with fixed limits
        self.ax2.cla()  # Clear the 2D image plot

        traj = np.array(self.camera_trajectory)
        
        # Plot camera trajectory
        self.ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='r', label='Camera Trajectory')

        # Plot previously seen landmarks in green
        # if self.previous_landmarks is not None:
        #     self.ax1.scatter(
        #         self.previous_landmarks[:, 0], 
        #         self.previous_landmarks[:, 1], 
        #         self.previous_landmarks[:, 2], 
        #         c='g', label='Previous Landmarks'
        #     )

        # Plot current landmarks in blue
        self.ax1.scatter(
            landmarks_3d[:, 0], 
            landmarks_3d[:, 1], 
            landmarks_3d[:, 2], 
            c='b', label='Current Landmarks'
        )

        # Update previous landmarks
        # self.previous_landmarks = landmarks_3d

        # Plot current camera pose
        self.ax1.scatter(camera_pose[0], camera_pose[1], camera_pose[2], c='r', s=10, label='Camera Pose')

        # Set axis labels
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()

        # Dynamically adjust the view to follow the camera
        self.ax1.view_init(elev=0, azim=270)
        self.ax1.set_xlim([camera_pose[0] - 15, camera_pose[0] + 15])
        self.ax1.set_ylim([camera_pose[1] - 15, camera_pose[1] + 15])
        self.ax1.set_zlim([camera_pose[2] - 15, camera_pose[2] + 15])

        # 2D Plot: Image and Keypoints
        self.ax2.imshow(image, cmap='gray')  # Image in grayscale
        self.ax2.scatter(keypoints[:, 0], keypoints[:, 1], c='g', marker='o', label='Keypoints')
        self.ax2.set_title('Processed Image with Keypoints')

        # 3D Plot without clearing
        #self.ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='r', label='Camera Trajectory')

        self.ax3.scatter(camera_pose[0], camera_pose[1], camera_pose[2], c='r', s=5, label='Camera Pose')
        self.ax3.set_xlabel('X')
        self.ax3.set_ylabel('Y')
        self.ax3.set_zlabel('Z')
        self.ax3.view_init(elev=0, azim=270)
        #self.ax3.legend()

        #   # Plot current landmarks in blue
        # self.ax3.scatter(
        #     landmarks_3d[:, 0], 
        #     landmarks_3d[:, 1], 
        #     landmarks_3d[:, 2], 
        #     c='b', label='Current Landmarks', s=2
        # )


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

        # Show the plot
        plt.draw()
        plt.pause(0.0001)  # Pause for a moment to update the plot