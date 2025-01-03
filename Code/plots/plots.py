import numpy as np
import glob
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt

# get the poses txt files in the folder
file_list = glob.glob('/Users/swissjab/Desktop/uzh_vision/VO_miniproject/Code/plots/pose*.txt')

#load the data from the txt files and differeniating between the different methods
data = {}
for file in file_list:
    if 'harris' in file:
        data['Harris'] = np.loadtxt(file)
    elif 'sift' in file:
        data['SIFT'] = np.loadtxt(file)
    elif 'shi_tomasi' in file:
        data['Shi Tomasi'] = np.loadtxt(file)

# get the ground truth poses
data["Ground Truth"] = np.loadtxt('/Users/swissjab/Desktop/uzh_vision/VO_miniproject/Datasets/parking/poses.txt')

# Reformat data[label] into 2 matrices, one for the rotation matrix and another one for the translation vector
rotation = {}
translation = {}
transform = {}
for label in data:
    rotation[label] = data[label][:, :9].reshape(-1, 3, 3)
    translation[label] = data[label][:, [3, 7, 11]].reshape(-1, 3, 1)
    transform[label] = data[label][:, :12].reshape(-1, 3, 4)

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set intrinsic matrix parameters
intrinsic_matrix = np.array([[331.37, 0, 320],
                             [0, 369.568, 240],
                             [0, 0, 1]])
sensor_size = (640, 480)
virtual_image_distance = 0.1

# Define colors for different labels
colors = {
    'Harris': 'r',
    'SIFT': 'g',
    'Shi Tomasi': 'b',
    'Ground Truth': 'k'
}

# Add legend
for label in colors:
    ax.plot([], [], color=colors[label], label=label)
ax.legend()

# Plot the camera poses
for label in transform:
    print(label, ": ", transform[label].shape[0])
    #for i in range(transform[label].shape[0]):
    for i in range(5):
        cam2world = np.eye(4)
        cam2world[:3, :4] = transform[label][i]
        pc.plot_camera(ax, cam2world=cam2world, M=intrinsic_matrix, sensor_size=sensor_size, virtual_image_distance=virtual_image_distance, color=colors[label], label=label if i == 0 else "")

# Add axis labels and title
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Camera Poses in 3D')

# Scale up the axes
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])

# Add legend
plt.show()
