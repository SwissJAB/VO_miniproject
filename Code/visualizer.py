import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class VisualOdometryVisualizer:
    def __init__(self, video_filename='output_video.mov', frame_size=(1920, 1080), fps=1.0):
        # Initialize the video writer with .mov format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec for .mov format
        self.out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
        
        # Store the frame size
        self.frame_size = frame_size
        
        # Set a fixed DPI for the figure (you can experiment with this value)
        self.dpi = 100  # Fixed DPI for the figure
        
        # Create a figure with the exact frame size (in inches)
        fig_width = self.frame_size[0] / self.dpi  # Width in inches
        fig_height = self.frame_size[1] / self.dpi  # Height in inches
        
        # Initialize the figure to match the frame size
        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        
        # Set up the axes for plotting
        self.ax1 = self.fig.add_subplot(121, projection='3d')  # 3D plot
        self.ax2 = self.fig.add_subplot(122)  # Image plot
        
        # Create a canvas to render the figure
        self.canvas = FigureCanvas(self.fig)

    def update_visualizations(self, landmarks_3d, camera_pose, image, keypoints):
        """
        Updates the visualizations and writes the frame to the video.

        Parameters:
        landmarks_3d: 3xN array of 3D landmarks
        camera_pose: camera pose [x, y, z]
        image: 2D array (image being processed)
        keypoints: 2xN array of keypoints (x, y)
        """
        # Clear the previous plots only if needed
        self.ax1.cla()  # Clear the 3D plot
        self.ax2.cla()  # Clear the 2D image plot

        self.ax1.scatter(landmarks_3d[0, :], landmarks_3d[1, :], landmarks_3d[2, :], c='b', label='Landmarks')
        self.ax1.scatter(-camera_pose[0], -camera_pose[1], -camera_pose[2], c='r', label='Camera Pose')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.legend()

        # Plot the image with keypoints in grayscale
        self.ax2.imshow(image, cmap='gray')  # Use cmap='gray' to display the image in grayscale
        self.ax2.scatter(keypoints[0, :], keypoints[1, :], c='g', marker='o', label='Keypoints')
        self.ax2.set_title('Processed Image with Keypoints')

        # Convert the plot to an image
        self.canvas.draw()

        # Extract the image from the canvas as a numpy array
        frame = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)

        # Get the canvas size from the figure (in pixels)
        width, height = self.fig.canvas.get_width_height()

        # Ensure the canvas size matches the video frame size
        if (height, width) != self.frame_size:
            print(f"Canvas size {width}x{height} doesn't match the video frame size. Resizing...")
            frame = frame.reshape((height, width, 3))
            frame_resized = cv2.resize(frame, self.frame_size)  # Resize only if needed
        else:
            frame = frame.reshape((height, width, 3))
            frame_resized = frame

        # Check if the frame size matches the expected size
        expected_size = self.frame_size[0] * self.frame_size[1] * 3
        if frame_resized.size != expected_size:
            print(f"Frame size mismatch: Expected {expected_size}, but got {frame_resized.size}")
            return
        
        # Write the resized frame to the video
        self.out.write(frame_resized)

    def close(self):
        """Releases the video writer and closes the figure."""
        self.out.release()
        cv2.destroyAllWindows()
