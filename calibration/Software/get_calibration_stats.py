import os
import json
import sys

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def main():

    if len(sys.argv) != 4:
        print(f"Usage: {sys.executable} {sys.argv[0]} <OpenMVG JSON> <Output folder> <Plot folder>")
        sys.exit(-1)

    input_omvg = sys.argv[1]
    output_folder = sys.argv[2]
    report_file = os.path.join(output_folder, 'calibration_stats.json')
    plot_folder = sys.argv[3]
    reproj_plot_folder = os.path.join(plot_folder, 'reprojection_error')
    os.makedirs(reproj_plot_folder, exist_ok=True)

    # Obtain the depths from openMVG calibration
    with open(input_omvg, 'r') as json_f:
        sfm_data = json.load(json_f)

    detected_points, reprojected_points, reproj_errors = check_reprojection_error(sfm_data)

    # Plot and process results
    plot_reprojected_points(reprojected_points, detected_points, reproj_plot_folder)
    process_reprojection_errors(reproj_errors, report_file, reproj_plot_folder)

# end main


def check_reprojection_error(sfm_data):

    # Dictionary with the extrinsics of each camera
    extrinsics = {}
    for pose in sfm_data["extrinsics"]:
        extrinsics[pose['key']] = {
            'R': np.array(pose["value"]["rotation"]),
            'c': np.array(pose["value"]["center"])
        }
    
    # Dictionary with the intrinsics of each camera
    intrinsics = {}
    for intrinsic in sfm_data["intrinsics"]:
        f = intrinsic["value"]["ptr_wrapper"]["data"]["focal_length"]
        c = intrinsic["value"]["ptr_wrapper"]["data"]["principal_point"]
        intrinsics[intrinsic['key']] = np.array([
            [f, 0, c[0]],
            [0, f, c[1]],
            [0, 0, 1]
        ])
            
    # Dictionary to store the reprojection errors
    detected_points = {}
    for key in extrinsics.keys():
        detected_points[key] = []
    reprojected_points = {}
    for key in extrinsics.keys():
        reprojected_points[key] = []
    reproj_errors = {}
    for key in extrinsics.keys():
        reproj_errors[key] = []

    print(f'Computing reprojection error for {len(sfm_data["structure"])} features...')

    # for each triangulated point
    for point in tqdm(sfm_data["structure"], colour='cyan'):

        # 3D point
        cloud_point = np.array(point["value"]["X"])

        # For each camera that 'observed' that point
        for observation in point["value"]["observations"]:


            camera_id = observation["key"]
            detected_point = observation["value"]["x"]

            # Reproject 3D point to camera space
            reproj_point= project_3d_to_2d(cloud_point, intrinsics[camera_id], extrinsics[camera_id])

            # Compute reprojection error
            error = eu_dis(reproj_point, detected_point)

            # Store the results
            detected_points[camera_id].append(detected_point)
            reprojected_points[camera_id].append(reproj_point)
            reproj_errors[camera_id].append(error)

        # for each observation

    # for each 3D point

    return detected_points, reprojected_points, reproj_errors
    
# check_reprojection_error


# Projects a 3D world point to a 2D image point using camera intrinsics and extrinsics.
def project_3d_to_2d(world_point, intrinsics, extrinsics):

    # Convert to numpy arrays
    K = intrinsics  # Intrinsic matrix (3x3)
    R = np.array(extrinsics['R'])  # Rotation matrix (3x3)
    c = np.array(extrinsics['c']).reshape(3, 1)  # Translation vector (3x1)
    P_w = np.array(world_point).reshape(3, 1)  # 3D point in world coordinates (3x1)

    # Transform world point to camera coordinates: P_c = R * P_w + t
    P_c = np.dot(R, (P_w - c))

    # Project onto image plane: p' = K * P_c
    p_homogeneous = np.dot(K, P_c)

    # Convert from homogeneous to 2D: (x, y) = (x'/w, y'/w)
    x = p_homogeneous[0, 0] / p_homogeneous[2, 0]
    y = p_homogeneous[1, 0] / p_homogeneous[2, 0]

    return (x, y)


def eu_dis(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    distance = np.linalg.norm(p1-p2)
    return distance


def plot_reprojected_points(reprojected_points, detected_points, plot_folder):

    for key in reprojected_points.keys():

        # format the lists as numpy arrays
        reproj = np.array(reprojected_points[key]).T
        detect = np.array(detected_points[key]).T

        plt.figure(figsize=(16, 9))
        plt.scatter(detect[0], detect[1], marker='o', c='blue', label='detected points', alpha=0.3)
        plt.scatter(reproj[0], reproj[1], marker='o', facecolors='none', edgecolors='red', label='reprojected points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Reprojected and detected points for cam {key}')
        plt.legend()
        plt.savefig(os.path.join(plot_folder, f'reprojection_points_{key}.png'))


def process_reprojection_errors(reproj_errors, report_file, plot_folder):

    print('Reprojection errors (measured in pixels):')
    # Compute the mean and standard deviation of the reprojection errors
    stats = {}
    for key in reproj_errors.keys():
        stats[key] = {
            'mean': np.mean(reproj_errors[key]),
            'std': np.std(reproj_errors[key]),
            'min': np.min(reproj_errors[key]),
            'max': np.max(reproj_errors[key]),
            'median': np.median(reproj_errors[key]),
            'q1': np.percentile(reproj_errors[key], 25),
            'q3': np.percentile(reproj_errors[key], 75)
        }
        print(f'Camera {key}: Mean = {stats[key]["mean"]:.2f}, Std = {stats[key]["std"]:.2f}')

    # Save the stats to a JSON file
    with open(report_file, 'w') as json_f:
        json.dump(stats, json_f, indent=4)

    # Boxplot of the reprojection errors
    plt.figure()
    plt.boxplot([reproj_errors[key] for key in reproj_errors.keys()])
    plt.xlabel('Camera')
    plt.ylabel('Reprojection error (pixels)')
    plt.title('Reprojection error boxplot')
    plt.xticks(range(1, len(reproj_errors)+1), [f'Cam {key}' for key in reproj_errors.keys()])
    plt.savefig(os.path.join(plot_folder, 'reprojection_error_boxplot.png'))

    # Create a histogram of the reprojection errors
    plt.figure()
    for key in reproj_errors.keys():
        plt.figure()
        plt.hist(reproj_errors[key], bins=32)
        plt.xlabel('Reprojection error (pixels)')
        plt.ylabel('Frequency')
        plt.title(f'Reprojection error histogram for cam {key}')
        plt.savefig(os.path.join(plot_folder, f'reprojection_error_{key}.png'))


if __name__ == '__main__':
    main()