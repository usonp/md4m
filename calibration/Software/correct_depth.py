import os
import json
import yaml
import sys

from tqdm import tqdm
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap

# import matplotlib
# matplotlib.rcParams.update({'font.size': 22})


def main():

    if len(sys.argv) != 8:
        print(f"Usage: {sys.executable} {sys.argv[0]} <OpenMVG JSON> <detected depth folder> <param folder> <num_corners_x> <num_corners_y> <output folder> <plot folder>")
        sys.exit(-1)

    input_omvg = sys.argv[1]
    detected_depth_folder = sys.argv[2]
    params_folder = sys.argv[3]
    num_corners_x = int(sys.argv[4])
    num_corners_y = int(sys.argv[5])
    output_fvv = sys.argv[6]
    plot_folder = os.path.join(sys.argv[7], 'depth_correction')

    ###############
    # Degree for the polynomial regression
    poly_degree = 1
    ###############

    # Create otput folders
    os.makedirs(output_fvv, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    num_corners_total = num_corners_x * num_corners_y
    print(f'Number of corners: {num_corners_total}')

    # Dictionary with lists of depth values for each camera
    detected_depth_dictionary = {}
    print(f'Processing Detected depth files from: {detected_depth_folder}')
    for filename in os.listdir(detected_depth_folder):
        if filename.endswith(".txt"):
            depth_detected_list = process_detected_depth(
                os.path.join(detected_depth_folder, filename),
                num_corners_total
            )
            id = int(filename.split('.')[0][-2:])
            detected_depth_dictionary[id]=depth_detected_list
        else:
            print(f'{filename} skipped')

    # Read Parameter file to get the serial number (yes, all those files just for this...)
    SN_dictionary = {}
    print(f'Obtaining serial numbers from: {params_folder}')
    for filename in os.listdir(params_folder):
        SN, id = process_param_file(os.path.join(params_folder, filename))
        SN_dictionary[id]=SN

    print(f'SNs: {SN_dictionary}')

    # Obtain the depths from openMVG calibration
    with open(input_omvg, 'r') as json_f:
        sfm_data = json.load(json_f)

    mvg_depth_dictionary, ratio_dictionary = get_mvg_depths(sfm_data, detected_depth_dictionary)

    # Perform regressions 
    print('Computing regressions for Depth correction')
    output_file = os.path.join(output_fvv,'DepthTransformationParameters.yaml')
    for camera_id in SN_dictionary.keys():

        detected_depth = np.array(detected_depth_dictionary[camera_id])
        mvg_depth = np.array(mvg_depth_dictionary[camera_id])
        ratio = np.array(ratio_dictionary[camera_id])
        SN = SN_dictionary[camera_id]

        # Generate array for coloring plot
        coloring_array, norm = generate_repeated_array(len(detected_depth), num_corners_total)

        # Remove all the zeros in the lists (they are invalid values)
        non_zero_mask = np.logical_and( (detected_depth > 0), (mvg_depth > 0) )
        detected_depth = detected_depth[non_zero_mask]
        mvg_depth = mvg_depth[non_zero_mask]
        ratio = ratio[non_zero_mask]
        coloring_array = coloring_array[non_zero_mask]


        print(f"Performing regression for camera {camera_id} with {len(detected_depth)} points")
        
        # cmap = gist_rainbow/viridis
        regression(detected_depth, mvg_depth, ratio, coloring_array, norm, SN, camera_id,
                   poly_degree, output_file, plot_folder, cmap_name='gist_rainbow')

# end main



def process_detected_depth(depth_detected_file, num_corners_total):

    depth_detected_list = [] # Track which frames were detected

    with open(depth_detected_file, 'r') as f:
        lines = f.readlines()

    index = 0  # Track current line
    expected_frame = 0  # Track current frame

    while index < len(lines):
        # Read camera and frame info
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        
        try:
            n_cam, n_frame = map(int, line.split())
        except ValueError:
            print(f"Skipping malformed line: {line}")
            index += 1
            continue

        # Check if the frame number is higher than expected (frame skipped)
        if n_frame > expected_frame:
            print(f"Camera {n_cam} skipped {n_frame - expected_frame} frame(s) from frame {expected_frame}")
            # Fill the list with 0s
            while (n_frame > expected_frame):
                for _ in range(num_corners_total):
                    depth_detected_list.append(0)
                expected_frame += 1

        expected_frame += 1
        
        # Move to the next line and read the depth values
        index += 1
        for _ in range(num_corners_total):
            if index >= len(lines):  
                print(f"Not enough points for Camera {n_cam}, Frame {n_frame}, skipping...")
                break

            line = lines[index].strip()
            if not line:
                index += 1
                continue

            try:
                depth = float(line)
                depth_detected_list.append(depth)
            except ValueError:
                print(f"Skipping malformed corner data: {line}")

            index += 1

    print(f'Number of points detected in {depth_detected_file}: {len(depth_detected_list)}')

    return depth_detected_list

#  process_detected_depth


def process_param_file(file):
    serial_number = None
    is_zed = False
    camera_number = None

    with open(file, 'r') as f:
        for line in f:
            if line.startswith("DEVICE:"):
                is_zed = "ZED" in line
            elif line.startswith("SERIAL NUMBER:"):
                serial_number = int(line.replace("SERIAL NUMBER: ", "").strip())
            elif line.startswith("Camera:"):
                camera_number = int(line.replace("Camera: ", "").strip())

    if is_zed:
        camera_number *= 2

    return serial_number, camera_number

# process_param_file


def get_mvg_depths(sfm_data, detected_depth_dictionary):

    # Initialize output dictionaries to lists of 0s
    len_dictionary = {key: len(value) for key, value in detected_depth_dictionary.items()}
    cameras_used = len_dictionary.keys()
    mvg_depths, ratios = ({key: [0] * value for key, value in len_dictionary.items()} for _ in range(2))

    # Create dictionary with the extrinsics of each camera
    extrinsics = {}
    for pose in sfm_data["extrinsics"]:
        extrinsics[pose['key']] = {
            'R': np.array(pose["value"]["rotation"]),
            'c': np.array(pose["value"]["center"])
        }

    print(f'Running depth study for {len(sfm_data["structure"])} points')

    # for each triangulated point
    for point in tqdm(sfm_data["structure"], colour='cyan'):

        # 3D point
        cloud_point = np.array(point["value"]["X"])

        # For each camera that 'observed' that point
        for observation in point["value"]["observations"]:

            camera_id = observation["key"]
            feature_id = observation["value"]["id_feat"]

            if camera_id not in cameras_used:
                # We are not using this camera for depth correction
                continue

            # Check list length to avoid out of bounds error
            current_length = len_dictionary[camera_id]
            if (current_length -1) < feature_id:
                # this camera does not have this feature
                print(f"[WARNING] - camera {camera_id} does not have this feature: current_length ({current_length -1}) < feature_id ({feature_id}")
                continue

            current_detected_depth = detected_depth_dictionary[camera_id][feature_id]

            if current_detected_depth == 0:
                # skip computations since detected depth is invalid
                print(f'DEPTH WAS ZERO cam: {camera_id} frame: {feature_id}')
                continue

            # Compute calib depth with reprojection
            translated_point = cloud_point - extrinsics[camera_id]['c']
            point_in_pose = np.dot(extrinsics[camera_id]['R'], translated_point)
            current_mvg_depth = point_in_pose[2]
            # current_mvg_depth = eu_dis(cloud_point,extrinsics[camera_id]['c'])
            current_ratio = current_mvg_depth / current_detected_depth

            # Store results
            mvg_depths[camera_id][feature_id] = current_mvg_depth
            ratios[camera_id][feature_id] = current_ratio
        
        # for each observation
    # for each 3D point

    return mvg_depths, ratios

# get_mvg_depths


def generate_repeated_array(length, repetitions):
    # Generate an array of repeated numbers
    num_repeats = length // repetitions
    remainder = length % repetitions
    array = np.repeat(np.arange(num_repeats), repetitions)
    # Add remaining repetitions if the length is not divisible by repetitions
    if remainder > 0:
        array = np.concatenate((array, np.repeat(num_repeats, remainder)))
    norm = plt.Normalize(vmin=array.min(), vmax=array.max())
    return array, norm


def regression( detected_depth, mvg_depth, ratio, coloring_array, norm, SN, id,
                deg, output_file, plot_folder, cmap_name='gist_rainbow'):
    
    # Ensure degree is between 1 and 4
    if deg < 1 or deg > 4:
        raise ValueError("Degree must be between 1 and 4.")
    
    cmap = get_cmap(cmap_name)
    
    # Perform regression using np.polyfit based on the degree
    coefficients = np.polyfit(detected_depth, ratio, deg)
    # coefficients = Poly.fit(detected_depth, ratio, deg=deg).convert().coef
    
    # Set variables to None by default
    a = b = c = d = e = None
    
    # Adjust based on the degree
    if deg == 1:
        b, c = coefficients  # Linear: y = b * x + c
    elif deg == 2:
        a, b, c = coefficients  # Quadratic: y = a * x^2 + b * x + c
    elif deg == 3:
        a, b, c, d = coefficients  # Cubic: y = a * x^3 + b * x^2 + c * x + d
    elif deg == 4:
        a, b, c, d, e = coefficients  # 4th-degree: y = a * x^4 + b * x^3 + c * x^2 + d * x + e
    
    # print(f"Coefficients: {coefficients}")
    
    # Create a plot of points and the regression curve
    plt.figure(figsize=(8, 6))
    plt.scatter(detected_depth, ratio, c=coloring_array, cmap=cmap, norm=norm, label='Data points')
    # plt.scatter(detected_depth, ratio, color='blue', label='Data points')

    # Generate the regression curve
    x_range = np.linspace(min(detected_depth), max(detected_depth), 500)
    
    if deg == 1:
        y_pred = b * x_range + c
    elif deg == 2:
        y_pred = a * x_range**2 + b * x_range + c
    elif deg == 3:
        y_pred = a * x_range**3 + b * x_range**2 + c * x_range + d
    elif deg == 4:
        y_pred = a * x_range**4 + b * x_range**3 + c * x_range**2 + d * x_range + e

    plt.plot(x_range, y_pred, color='red', label=f'{deg}-degree regression curve')

    plt.title(f'{deg}-Degree Polynomial Regression for camera {id} ({SN})')
    plt.xlabel('Z_detected')
    plt.ylabel('Z_mvg/Z_detected')
    plt.legend()
    plt.savefig(os.path.join(plot_folder, f'{id}_{SN}.png'))

    # Read the existing YAML file (if it exists) or create a new dictionary
    if os.path.exists(output_file):
        with open(output_file, 'r') as yaml_file:
            data_to_save = yaml.safe_load(yaml_file) or {}
    else:
        data_to_save = {}

    # Add a new key for the current SN and save the coefficients
    data_to_save[str(SN)] = {}
    
    if deg == 1:
        data_to_save[str(SN)]['slope'] = float(b)
        data_to_save[str(SN)]['offset'] = float(c)
    if deg >= 2:
        data_to_save[str(SN)]['a'] = float(a)
        data_to_save[str(SN)]['b'] = float(b)
        data_to_save[str(SN)]['c'] = float(c)
    if deg >= 3:
        data_to_save[str(SN)]['d'] = float(d)
    if deg == 4:
        data_to_save[str(SN)]['e'] = float(e)

    # Step 7: Write the updated data back to the YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(data_to_save, yaml_file)
    
    print(f"YAML file {output_file} updated with SN {SN}")

    # Additionally, crate a plot with the linear values
    plt.figure(figsize=(12, 9))
    y_pred2 = x_range * y_pred
    plt.scatter(detected_depth, mvg_depth, c=coloring_array, cmap=cmap, norm=norm, label='Data points')
    # plt.scatter(detected_depth, mvg_depth, color='blue', label='Data points')
    plt.plot(x_range, y_pred2, color='red', linewidth=2, label=f'{deg+1}-degree regression curve')
    # plt.title(f'Estimated depth vs calibrated depth for camera {id} ({SN})')
    plt.xlabel('Estimated depth')
    plt.ylabel('Calibration depth (mm)')
    # plt.ylim([0,4000])
    
    # plt.scatter(mvg_depth, detected_depth, color='blue', label='Data points')
    # plt.plot(y_pred2, x_range, color='red', label=f'{deg+1}-degree regression curve')
    # plt.title(f'Detected depth vs OpenMVG depth for camera {id} ({SN})')
    # plt.xlabel('Z calibration (mm)')
    # plt.ylabel('Z estimated')
    # plt.ylim([-50,50])

    plt.legend()
    plt.savefig(os.path.join(plot_folder, f'{id}_{SN}_lineal.png'))

    pearson_corr, _ = pearsonr(detected_depth, mvg_depth)
    spearman_corr, _ = spearmanr(detected_depth, mvg_depth)
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")


def eu_dis(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    distance = np.linalg.norm(p1-p2)
    return distance


if __name__ == '__main__':
    main()