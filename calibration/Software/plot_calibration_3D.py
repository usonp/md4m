import json
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap


# Load OpenMVG JSON file
def load_openmvg_json(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    with open(filepath, "r") as f:
        return json.load(f)

# Extract camera positions and orientations
def get_camera_data(sfm_data):
    cameras = []
    rotations = []
    focals = []
    c_xs = []
    c_ys = []
    
    for cam in sfm_data.get("extrinsics", []):
        C = np.array(cam["value"]["center"])  # Camera position
        R = np.array(cam["value"]["rotation"])  # Rotation matrix
        cameras.append(C)
        rotations.append(R)

    for cam in sfm_data.get("intrinsics", []):
        f = cam["value"]["ptr_wrapper"]["data"]["focal_length"]
        c = cam["value"]["ptr_wrapper"]["data"]["principal_point"]
        focals.append(f)
        c_xs.append(c[0])
        c_ys.append(c[1])
    
    return np.array(focals), np.array(c_xs), np.array(c_ys), np.array(cameras), np.array(rotations) 

# Extract 3D points from structure
def get_triangulated_points(sfm_data):
    return np.array([np.array(pt["value"]["X"]) for pt in sfm_data.get("structure", [])])

# Convert OpenMVG's Y-up coordinate system to Matplotlib's Z-up
def convert_coordinates(coords):
    S = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0,  1,  0]
    ])
    result = []

    for coord in coords:
        result.append((S @ coord)) # Swap Y and Z
    
    return np.array(result)

# Draw correctly oriented camera frustums
#   - Z0: Depth at which the image plane is drawn
def draw_camera_frustum(ax, cam_id, f, c_x, c_y, R, C, Z0, color='r'):

    # Define image plane corner points in normalized image coordinates
    img_corners = np.array([
        [c_x, c_y, f],   # Top-right
        [c_x, -c_y, f],  # Bottom-right
        [-c_x, -c_y, f], # Bottom-left
        [-c_x, c_y, f],  # Top-left
    ]).T

    # Scale to the given depth
    img_corners *= Z0 / f

    # Convert to world coordinates using rotation and translation
    world_corners = R.T @ img_corners + C.reshape(3, 1)

    # Define camera center
    camera_center = C.reshape(3, 1)

    # Draw camera frustum edges
    edges = [
        (camera_center, world_corners[:, i:i+1]) for i in range(4)
    ] + [
        (world_corners[:, i:i+1], world_corners[:, (i+1) % 4:i+2]) for i in range(4)
    ]
    
    for start, end in edges:
        ax.plot(*np.hstack((start, end)), color=color)

    # Add text with camera ID
    ax.text(C[0], C[1], C[2], f'Cam {cam_id}', fontsize=12, weight='bold')


# Draw camera vectors (Up and Forward)
def draw_camera_vectors(ax, C, R, scale=0.1):
    """Draws up and forward vectors for a camera."""
    
    # Get the up vector (originally Y-axis in OpenMVG, now mapped to Z)
    up_vector = R[1, :] * scale  # Second column of R is the Y-axis (now our Z-axis)

    # Get the forward vector (originally Z-axis in OpenMVG, now mapped to Y)
    forward_vector = R[2, :] * scale  # Third column of R is the Z-axis (now our Y-axis)

    # Convert coordinates
    up_vector = convert_coordinates(np.array([up_vector]))[0]
    forward_vector = convert_coordinates(np.array([forward_vector]))[0]

    # Draw vectors
    ax.quiver(C[0], C[1], C[2], up_vector[0], up_vector[1], up_vector[2], color='blue', length=scale, label="Up Vector")
    ax.quiver(C[0], C[1], C[2], forward_vector[0], forward_vector[1], forward_vector[2], color='red', length=scale, label="Forward Vector")


def set_axes_equal(ax):
    """Make the 3D axes have equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    center = np.mean(limits, axis=1)
    max_range = np.max(limits[:, 1] - limits[:, 0]) / 2

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)


# Plot the cameras and 3D points
def plot_scene(focals, c_xs, c_ys, cameras, rotations, points, num_corners=50):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Convert coordinate system (swap Y and Z)
    # cameras = convert_coordinates(cameras)
    # rotations = convert_coordinates(rotations)
    # points = convert_coordinates(points)

    # Plot cameras
    for i in range(len(cameras)):
        # draw_camera_vectors(ax, cameras[i], rotations[i], scale=20)
        draw_camera_frustum(ax, i, focals[i], c_xs[i], c_ys[i], rotations[i], cameras[i], 250)

    # Plot triangulated points with colormap
    if len(points) > 0:
        num_points = len(points)
        cmap = get_cmap('viridis', num_points // num_corners + 1)  # Generate colors for groups of num_corners
        for i in range(0, num_points, num_corners):
            color = cmap(i // num_corners)  # Get color from colormap
            batch = points[i:i+num_corners]  # Get num_corners points
            ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2], c=[color]*len(batch), s=5)  # Assign color
    else:
        print("Warning: No 3D points found!")

    # Set equal scale for all axes
    set_axes_equal(ax)

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("OpenMVG 3D Reconstruction")
    # ax.legend()
    ax.view_init(elev=50, azim=150, roll=0-114)
    plt.show()

# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_sfm_data.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    sfm_data = load_openmvg_json(filepath)

    focals, c_xs, c_ys, cameras, rotations = get_camera_data(sfm_data)
    points = get_triangulated_points(sfm_data)

    plot_scene(focals, c_xs, c_ys, cameras, rotations, points)
