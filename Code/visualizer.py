import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class VisualOdometryVisualizer:
    def __init__(self):
        # Initialize the figure
        self.fig = plt.figure(figsize=(12, 6))

        # Set up the axes for plotting
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # 3D plot
        self.ax2 = self.fig.add_subplot(122)  # 2D image plot

        # Define fixed axis limits for the 3D plot
        self.ax1.set_xlim([-50, 50])  # Set X-axis limit
        self.ax1.set_ylim([-50, 50])  # Set Y-axis limit
        self.ax1.set_zlim([-50, 50])  # Set Z-axis limit

    def update_visualizations(self, landmarks_3d, camera_pose, image, keypoints):
        """
        Updates the figure with 3D landmarks and 2D keypoints.
        
        Parameters:
        landmarks_3d: 3xN array of 3D landmarks
        camera_pose: camera pose [x, y, z]
        image: 2D array (image being processed)
        keypoints: 2xN array of keypoints (x, y)
        """
        # Clear previous plots
        self.ax1.cla()  # Clear the 3D plot
        self.ax2.cla()  # Clear the 2D image plot

        # 3D Plot: Landmarks and Camera Pose
        self.ax1.scatter(landmarks_3d[0, :], landmarks_3d[1, :], landmarks_3d[2, :], c='b', label='Landmarks')
        self.ax1.scatter(-camera_pose[0], -camera_pose[1], -camera_pose[2], c='r', label='Camera Pose')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()

        # Set the axis limits to prevent rescaling
        self.ax1.set_xlim([-10, 60])  # Keep fixed limits for X-axis
        self.ax1.set_ylim([-50, 50])  # Keep fixed limits for Y-axis
        self.ax1.set_zlim([-10, 10])  # Keep fixed limits for Z-axis

        # 2D Plot: Image and Keypoints
        self.ax2.imshow(image, cmap='gray')  # Image in grayscale
        self.ax2.scatter(keypoints[0, :], keypoints[1, :], c='g', marker='o', label='Keypoints')
        self.ax2.set_title('Processed Image with Keypoints')

        # Show the plot
        plt.draw()
        plt.pause(0.001)  # Pause for a moment to update the plot

# Example usage
# visualizer = VisualOdometryVisualizer()
# visualizer.update_figure(landmarks_3d, camera_pose, image, keypoints)
