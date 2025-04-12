import os
import sys
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    """Main function to handle command-line arguments and execute processing."""
    if len(sys.argv) < 5:
        print(f"Usage: python {sys.argv[0]} <input_file> <output_plot> <num_corners_x> <num_corners_y> <square_size>")
        sys.exit(1)

    sfm_data_file = sys.argv[1]
    output_plot_folder = os.path.join(sys.argv[2], 'scale')
    num_corners_x = int(sys.argv[3])
    num_corners_y = int(sys.argv[4])
    square_size = float(sys.argv[5])

    with open(sfm_data_file) as f:
        sfm_data = json.load(f)

    ##################################
    # Older versions of openCV detect the corners in vertical, newer do it in horizontal
    points_detected_in_horizontal = True
    ##################################

    # Read points from the file
    points = read_points(sfm_data)

    # Separate points into groups of patterns
    num_corners_total = num_corners_x * num_corners_y
    print('Number of corners: ', num_corners_total)
    print('Number of triangulated corners:', len(points))
    # assert len(points) % num_corners_total == 0

    distances_x, distances_y = compute_distances(points, num_corners_x, num_corners_y, points_detected_in_horizontal)

    # Compute metrics
    print('Metrics for X:')
    _, x_std, _ = compute_metrics(distances_x)
    print('Metrics for Y:')
    _, y_std, _ = compute_metrics(distances_y)

    ## Remove outliers:
    # Some points of the board may be missing, so it is very difficult to decide how to compute the distances
    # The approach used here is based on computing the distances as if there was not any error
    # then, using the std, the incorrect measures are filtered
    alpha = 2
    if points_detected_in_horizontal:
        threshold = alpha * x_std
    else:
        threshold = alpha * y_std
    distances = np.concatenate((distances_x,distances_y), axis=0)
    clean_distances = distances[distances < threshold]
    print('Filtered metrics:')
    mean, _, _ = compute_metrics(clean_distances)

    # Generate and save the plot
    os.makedirs(output_plot_folder, exist_ok=True)
    plot_distance_info(distances_x, distances_y, clean_distances, output_plot_folder, threshold)


    # Scale computed
    scale = square_size/mean
    print('Final scale: ', scale)

    # Rescale the calibration
    base, ext = os.path.splitext(sfm_data_file)
    output_file =  f"{base}_scaled{ext}"
    rescale_calibration(sfm_data, scale, output_file)


# main

def read_points(sfm_data):
    points_3D = []
    num_points = len(sfm_data["structure"])
    key = 0
    while len(points_3D) != num_points:
        key_not_found = True
        for coords in sfm_data["structure"]:
            # print(key)
            if key == coords["key"]:
                x_point = coords["value"]["X"][0]
                y_point = coords["value"]["X"][1]
                z_point = coords["value"]["X"][2]
                point_3D = (x_point, y_point, z_point)
                points_3D.append(point_3D)
                key +=1
                key_not_found = False
        if key_not_found:
            # print(f'Key {key} not found')
            key +=1

    return points_3D


def compute_distances(points, num_corners_x, num_corners_y, points_detected_in_horizontal):
    dists_x = []
    dists_y = []

    if points_detected_in_horizontal:
        corners_to_skip = num_corners_x
        other_corners = num_corners_y
    else:
        corners_to_skip = num_corners_y
        other_corners = num_corners_x

    check_last_row = corners_to_skip * (other_corners -1)
    num_corners_total = num_corners_x * num_corners_y

    for i in range(len(points)-1):
        # Distance to adjacent points
        dists_y.append(eu_dis(points[i], points[i+1]))

        # Distance to points on other row/column
        if (i + corners_to_skip) < (len(points) -1): # prevent out of bounds
            if (i % num_corners_total) < check_last_row:
                dists_x.append(eu_dis(points[i], points[i+corners_to_skip]))

    if points_detected_in_horizontal:
        return dists_y, dists_x
    else:
        return dists_x, dists_y


def eu_dis(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    distance = np.linalg.norm(p1-p2)
    return distance


def compute_metrics(distances):
    """Computes mean, standard deviation, and variance of distances."""
    mean_dist = np.mean(distances)
    std_dev = np.std(distances)
    variance = np.var(distances)

    print(f"Mean Distance: {mean_dist:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Variance: {variance:.4f}")

    return mean_dist, std_dev, variance


def plot_distance_info(dists_x, dists_y, clean_dists, output_path, threshold, bins=24):

    # Plot distance in x info
    sns.histplot(dists_x, kde=False, bins=bins)
    plt.axvline(threshold, color='red', linewidth=2)
    plt.title('Distance distribution in the X-axis')
    plt.xlabel('Distance value')
    plt.ylabel('Density')
    plt.savefig(f'{output_path}/distance_distribution_x.png')
    plt.close()

    plt.plot(dists_x)
    plt.title('Distance in the X-axis')
    plt.ylabel('Distance')
    plt.savefig(f'{output_path}/distance_x.png')
    plt.close()

    # Plot distance in y info
    sns.histplot(dists_y, kde=True, bins=bins)
    plt.axvline(threshold, color='red', linewidth=2)
    plt.title('Distance distribution in the Y-axis')
    plt.xlabel('Distance value')
    plt.ylabel('Density')
    plt.savefig(f'{output_path}/distance_distribution_y.png')
    plt.close()

    plt.plot(dists_y)
    plt.title('Distance in the Y-axis')
    plt.ylabel('Distance')
    plt.savefig(f'{output_path}/distance_y.png')
    plt.close()

    # Plot distance used for the scale computation
    sns.histplot(clean_dists, kde=True, bins=bins)
    plt.axvline(np.mean(clean_dists), color='red', linewidth=2, label='Selected value')
    plt.title('Distance distribution used for scale computation')
    plt.xlabel('Distance value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{output_path}/distance_distribution_scale.png')
    plt.close()

    plt.plot(clean_dists)
    plt.title('Distance used for scale computation')
    plt.ylabel('Distance')
    plt.savefig(f'{output_path}/distance_scale.png')
    plt.close()


def rescale_calibration(sfm_data, scale, output_file):
    print('Rescaling calibration...')
    poses = range(len(sfm_data["extrinsics"]))
    for pose in poses:
        coords = range(len(sfm_data["extrinsics"][pose]["value"]["center"]))
        for coord in coords:
            scaled_pose = float(scale)*float(sfm_data["extrinsics"][pose]["value"]["center"][coord])
            sfm_data["extrinsics"][pose]["value"]["center"][coord] = float(scaled_pose)

    # Rescale the points coordinates
    points = range(len(sfm_data["structure"]))
    for point in points:
        coordinates = range(len(sfm_data["structure"][point]["value"]["X"]))
        for coordinate in coordinates:
            scaled_point = float(scale)*float(sfm_data["structure"][point]["value"]["X"][coordinate])
            sfm_data["structure"][point]["value"]["X"][coordinate] = float(scaled_point)

    with open(output_file, 'w') as rs:
        json.dump(sfm_data, rs, indent=4)
    rs.close
    print(f'Rescaled calibration saved as {output_file}')



if __name__ == "__main__":
    main()
