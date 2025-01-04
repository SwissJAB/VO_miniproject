import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class VisualOdometryVisualizer:
    def __init__(self):
        # Initialize the figure
        self.fig = plt.figure(figsize=(12, 6))

        # Set up the axes for plotting
        self.ax1 = self.fig.add_subplot(141, projection='3d')  # 3D plot
        self.ax2 = self.fig.add_subplot(142)  # 2D image plot
        self.ax3 = self.fig.add_subplot(143)  # 2D image plot
        self.ax4 = self.fig.add_subplot(144)  # 2D image plot

        # Define fixed axis limits for the 3D plot
        self.ax1.set_xlim([-5, 15])  # Set X-axis limit
        self.ax1.set_ylim([-5, 15])  # Set Y-axis limit
        self.ax1.set_zlim([-5, 15])  # Set Z-axis limit

        self.camera_trajectory = []
        self.n_values1 = {
            "n_X": [], "n_P": []
        }
        self.n_values2 = {
            "n_T": [], "n_C": [], "n_F": []
        }

    def update_visualizations(self, landmarks_3d, camera_pose, image, keypoints, n_X, n_P, n_T, n_C, n_F):
        """
        Updates the figure with 3D landmarks and 2D keypoints.
        
        Parameters:
        landmarks_3d: 3xN array of 3D landmarks
        camera_pose: camera pose [x, y, z]
        image: 2D array (image being processed)
        keypoints: 2xN array of keypoints (x, y)
        """
        self.camera_trajectory.append(camera_pose)

        self.n_values1["n_X"].append(n_X)
        self.n_values1["n_P"].append(n_P)
        self.n_values2["n_T"].append(n_T)
        self.n_values2["n_C"].append(n_C)
        self.n_values2["n_F"].append(n_F)
        
        # Clear previous plots
        self.ax1.cla()  # Clear the 3D plot
        self.ax2.cla()  # Clear the 2D image plot
        self.ax3.cla()

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

        # Update previous landmarks
        # self.previous_landmarks = landmarks_3d

        # Plot current camera pose
        self.ax1.scatter(camera_pose[0], camera_pose[1], camera_pose[2], c='r', s=10, label='Camera Pose')

        # Set axis labels
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        #self.ax1.legend()

        # Dynamically adjust the view to follow the camera
        self.ax1.view_init(elev=0, azim=90)
        self.ax1.set_xlim([camera_pose[0] - 5, camera_pose[0] + 15])
        self.ax1.set_ylim([camera_pose[1] - 5, camera_pose[1] + 15])
        self.ax1.set_zlim([camera_pose[2] - 5, camera_pose[2] + 15])

        # 2D Plot: Image and Keypoints
        self.ax2.imshow(image, cmap='gray')  # Image in grayscale
        self.ax2.scatter(keypoints[:, 0], keypoints[:, 1], c='g', marker='o', label='Keypoints')
        self.ax2.set_title('Processed Image with Keypoints')

        time_steps = range(len(self.n_values1["n_X"]))
        self.ax3.plot(time_steps, self.n_values1["n_X"], label="n_X")
        self.ax3.plot(time_steps, self.n_values1["n_P"], label="n_P")
        self.ax3.set_title("Values Over Time")
        self.ax3.set_xlabel("Time Step")
        self.ax3.set_ylabel("Value")
        self.ax3.legend()

        self.ax4.plot(time_steps, self.n_values2["n_T"], label="n_T")
        self.ax4.plot(time_steps, self.n_values2["n_C"], label="n_C")
        self.ax4.plot(time_steps, self.n_values2["n_F"], label="n_F")
        self.ax4.set_title("Values Over Time")
        self.ax4.set_xlabel("Time Step")
        self.ax4.set_ylabel("Value")
        # Show the plot
        plt.draw()
        plt.pause(0.001)  # Pause for a moment to update the plot