import os
import json
import sys

from tqdm import tqdm
import numpy as np


def main():

    if len(sys.argv) != 6:
        print(f"Usage: {sys.executable} {sys.argv[0]} <OpenMVG JSON> <detected corners folder> <detected depth folder> <num_corners_x> <num_corners_y>")
        sys.exit(-1)

    input_omvg = sys.argv[1]
    detected_corners_folder = sys.argv[2]
    detected_depth_folder = sys.argv[3]
    num_corners_x = int(sys.argv[4])
    num_corners_y = int(sys.argv[5])

    ################
    verbose = False
    ################

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

    # Dictionary with lists of corners for each camera
    detected_corners_dictionary = {}
    print(f'Processing Detected corners files from: {detected_corners_folder}')
    for filename in os.listdir(detected_corners_folder):
        if filename.endswith(".txt"):
            corners_detected_list = process_chessboard_corners(
                os.path.join(detected_corners_folder, filename),
                num_corners_x, num_corners_y
            )
            id = int(filename.split('.')[0][-2:])
            detected_corners_dictionary[id]=corners_detected_list
        else:
            print(f'{filename} skipped')

    check_calibration_files(detected_corners_dictionary, detected_depth_dictionary, verbose=verbose)

    # Obtain the depths from openMVG calibration
    with open(input_omvg, 'r') as json_f:
        sfm_data = json.load(json_f)

    check_mvg_stuff(sfm_data, detected_corners_dictionary, detected_depth_dictionary, verbose=verbose)

# end main


def process_detected_depth(depth_detected_file, num_corners_total):

    depth_detected_list = [] # Track which features were detected

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


def process_chessboard_corners(file_path, num_corners_x, num_corners_y, verbose=False):
    
    corners_detected_list = [] # Track which frames were detected

    with open(file_path, 'r') as f:
        lines = f.readlines()

    index = 0  # Track current line
    expected_frame = 0  # Track current frame
    num_total_corners = num_corners_x * num_corners_y
    filename = os.path.basename(file_path).split('.')[0]

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
            # Add False to the detected list and add zeros to the descriptor file
            while (n_frame > expected_frame):
                for _ in range(num_total_corners):
                    corners_detected_list.append((0,0))
                expected_frame += 1
        if verbose:
            print(f'Processing Camera {n_cam}, Frame {n_frame}, expected {expected_frame}')
        expected_frame += 1
        
        # Move to the next line and read the corners
        index += 1

        for _ in range(num_total_corners):
            if index >= len(lines):  
                print(f"Not enough points for Camera {n_cam}, Frame {n_frame}, skipping...")
                break

            line = lines[index].strip()
            if not line:
                print('NOT LINE')
                index += 1
                continue

            try:
                x, y = map(float, line.split(', '))
                corners_detected_list.append((x, y))
            except ValueError:
                print(f"Skipping malformed corner data: {line}")

            index += 1

    print(f"{filename} processed")

    return corners_detected_list

# process_chessboard_corners


def check_calibration_files(detected_corners_dictionary, detected_depth_dictionary, verbose=False):

    # Are the files consistent?
    print('\nChecking consistency between detected corners and depths...')
    for camera_id in detected_corners_dictionary.keys():
        len_corners = len(detected_corners_dictionary[camera_id])
        len_depths = len(detected_depth_dictionary[camera_id])
        if len_corners != len_depths:
            print(f'[ERROR] - Camera {camera_id} has different number of corners ({len_corners}) and depths ({len_depths})')
        else:
            print(f'[SUCCESS] - Camera {camera_id} has the same number of corners ({len_corners}) and depths ({len_depths})')

    # Check each feature
    correct_depth_bad_corner = 0
    correct_corner_bad_depth = 0
    missing_features = 0
    print('\nChecking consistency between each detected corner and depth...')
    for camera_id in detected_corners_dictionary.keys():
        for i in range(len(detected_corners_dictionary[camera_id])):
            corner = detected_corners_dictionary[camera_id][i]
            depth = detected_depth_dictionary[camera_id][i]
            if corner == (0,0) and depth != 0:
                if verbose:
                    print(f'[ERROR] - Camera {camera_id} feature {i} has depth {depth} but no corners detected')
                correct_depth_bad_corner += 1
            elif corner != (0,0) and depth == 0:
                if verbose:
                    print(f'[ERROR] - Camera {camera_id} feature {i} has corners {corner} but no depth detected')
                correct_corner_bad_depth += 1
            elif corner == (0,0) and depth == 0:
                if verbose:
                    print(f'[WARNING] - Camera {camera_id} feature {i} has no corners or depth detected')
                missing_features += 1
    # Report errors
    print(f'Errors studied:')
    print(f'correct_depth_bad_corner: {correct_depth_bad_corner}')
    print(f'correct_corner_bad_depth: {correct_corner_bad_depth}')
    if correct_depth_bad_corner + correct_corner_bad_depth == 0:
        print('[SUCCESS]- No errors found! Calibration files are consistent')
    else:    
        print('[ERRORS FOUND]- Please check the errors above')
    print(f'missing_features: {missing_features}')

# check_calibration_files


def check_mvg_stuff(sfm_data, detected_corners_dictionary, detected_depth_dictionary, verbose=False):

    # Initialize output dictionaries to lists of 0s
    len_dictionary = {key: len(value) for key, value in detected_depth_dictionary.items()}
    cameras_used = len_dictionary.keys()

    # Errors to track
    feature_out_of_range = 0
    invalid_depth = 0
    invalid_corners = 0
    feature_coords_not_matching = 0
    missing_features = 0

    print(f'\nLooking for errors on {len(sfm_data["structure"])} features...')
    current_feature = 0

    sorted_features = sorted(sfm_data["structure"], key=lambda x: x["key"])

    # for each triangulated point
    for point in tqdm(sorted_features, colour='cyan'):

        mvg_feature = point["key"]
        if current_feature != mvg_feature:
            if verbose:
                print(f'[mvg_feature_mismatch] - current_feature: {current_feature} != mvg_feature: {mvg_feature}')
            missing_features += mvg_feature - current_feature
            current_feature = mvg_feature

        # For each camera that 'observed' that point
        for observation in point["value"]["observations"]:

            camera_id = observation["key"]
            feature_id = observation["value"]["id_feat"]
            current_coords = observation["value"]["x"]
            feature_x = current_coords[0]
            feature_y = current_coords[1]

            if camera_id not in cameras_used:
                # We are not using this camera for depth correction
                continue

            # Check list length to avoid out of bounds error
            current_length = len_dictionary[camera_id]
            if (current_length -1) < feature_id:
                # this camera does not have this feature
                if verbose:
                    print(f"[feature_out_of_range] - camera {camera_id} current_length ({current_length -1}) < feature_id ({feature_id}")
                feature_out_of_range += 1
                continue

            current_detected_depth = detected_depth_dictionary[camera_id][feature_id]
            current_detected_corners = detected_corners_dictionary[camera_id][feature_id]
            current_x = current_detected_corners[0]
            current_y = current_detected_corners[1]

            if current_detected_depth == 0:
                print(f'[invalid_depth] - camera: {camera_id} feature: {feature_id}')
                invalid_depth += 1

            if current_x == 0 and current_y == 0:
                if verbose:
                    print(f'[invalid_corners] - camera: {camera_id} feature: {feature_id}')
                invalid_corners += 1

            if eu_dis((feature_x, feature_y), (current_x, current_y)) > 1:
                if verbose:
                    print(f'[feature_coords_not_matching] - camera: {camera_id} feature: {feature_id} - ({feature_x}, {feature_y}) != ({current_x}, {current_y})')
                feature_coords_not_matching += 1
        
        # for each observation

        current_feature += 1

    # for each 3D point

    # Report errors
    print(f'Errors studied:')
    print(f'feature_out_of_range: {feature_out_of_range}')
    print(f'invalid_depth: {invalid_depth}')
    print(f'invalid_corners: {invalid_corners}')
    print(f'feature_coords_not_matching: {feature_coords_not_matching}')

    if feature_out_of_range + invalid_depth + invalid_corners + feature_coords_not_matching == 0:
        print('[SUCCESS]- No errors found! Everything is going according to plan...')
    else:
        print('[ERRORS FOUND]- Please check the errors above')
    print(f'missing_features: {missing_features}')

# check_mvg_stuff


def eu_dis(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    distance = np.linalg.norm(p1-p2)
    return distance


if __name__ == '__main__':
    main()