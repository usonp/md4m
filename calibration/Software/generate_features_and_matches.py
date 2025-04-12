import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt


def main():

    ##############
    verbose = False
    ##############

    if len(sys.argv) < 5:
        print(f"Usage: python {sys.argv[0]} <input_folder> <feature_folder> <num_corners_x> <num_corners_y> <plot_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    feature_folder = sys.argv[2]
    num_corners_x = int(sys.argv[3])
    num_corners_y = int(sys.argv[4])
    plot_folder = os.path.join(sys.argv[5], 'detected_points')

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    os.makedirs(feature_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    # Store lists of boleans in a dictionary to track the frames found
    frames_detected_dictionary = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            frames_detected_list = process_chessboard_corners(
                file_path=os.path.join(input_folder, filename),
                features_path=feature_folder,
                num_corners_x=num_corners_x,
                num_corners_y=num_corners_y,
                plot_folder=plot_folder,
                verbose=verbose
            )
            id = int(filename.split('.')[0][-2:])
            frames_detected_dictionary[id]=frames_detected_list
        else:
            print(f'{filename} skipped')

    num_corners_total = num_corners_x * num_corners_y
    generate_matches_file(feature_folder, frames_detected_dictionary, num_corners_total)

# main


def process_chessboard_corners(file_path, features_path,
                               num_corners_x, num_corners_y,
                               plot_folder,
                               bytes_per_corner=128,
                               verbose=False):
    
    all_points = []  # Store all detected corners for plotting
    frames_detected_list = [] # Track which frames were detected

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
                frames_detected_list.append(False)
                with open(f'{features_path}/{filename}.feat', 'a') as f:
                    for _ in range(num_total_corners):
                        f.write('0 0 1 0\n')
                expected_frame += 1
        if verbose:
            print(f'Processing Camera {n_cam}, Frame {n_frame}, expected {expected_frame}')
        expected_frame += 1
        
        # Move to the next line and read the corners
        index += 1
        corners = []

        for _ in range(num_total_corners):
            if index >= len(lines):  
                print(f"Not enough points for Camera {n_cam}, Frame {n_frame}, skipping...")
                break

            line = lines[index].strip()
            if not line:
                index += 1
                continue

            try:
                x, y = map(float, line.split(', '))
                corners.append((x, y))
                all_points.append((x, y))
            except ValueError:
                print(f"Skipping malformed corner data: {line}")

            index += 1

        # Save extracted corners to a .feat file
        with open(f'{features_path}/{filename}.feat', 'a') as f:
            for x, y in corners:
                f.write(f'{x} {y} 1 0\n')

        # Mark the frame as detected
        frames_detected_list.append(True)

    # Convert to numpy array for easier plotting
    all_points = np.array(all_points)

    # Plot all extracted points
    if all_points.size > 0:
        plt.figure(figsize=(16, 9))
        # plt.scatter(all_points[:, 0], all_points[:, 1], marker='o', color='blue')
        # plt.scatter(all_points[:, 0], all_points[:, 1], marker='o', facecolors='none', edgecolors='blue')
        plt.scatter(all_points[:, 0], all_points[:, 1], marker='o', color='blue', alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"Detected Corners for {filename}")
        plt.savefig(os.path.join(plot_folder, f"{filename}_points.png"), dpi=300)

    # Generate a descriptor file with random bytes
    desc_file = f'{features_path}/{filename}.desc'
    # num_bytes = all_points.shape[0]*bytes_per_corner
    # generate_binary_file(desc_file, num_bytes)
    num_bytes = len(frames_detected_list) * num_total_corners
    generate_fake_desc_file(desc_file, num_bytes, bytes_per_corner)


    print(f"{filename} processed")

    return frames_detected_list




# Function to generate random bytes and save them to a binary file
def generate_binary_file(file_path, num_bytes):
    # Generate random bytes
    random_bytes = bytearray(random.getrandbits(8) for _ in range(num_bytes))

    # Write the random bytes to the file
    with open(file_path, 'wb') as f:
        f.write(random_bytes)
    print(f"Descriptor file generated at {file_path} with {num_bytes} bytes.")

# Specify the output file and the number of bytes
output_file = 'random_data.bin'
num_bytes = 100  # Change this to the number of bytes you want

# Generate the binary file


def generate_fake_desc_file(filename, num_descriptors, descriptor_size=128):
    byte_distribution = [
        (0x00, 40), (0x21, 10), (0x24, 10), (0x22, 8), 
        (0x15, 7), (0x23, 7), (0x18, 6), (0x30, 5),
        (0x3F, 5), (0x1A, 4), (0x10, 4), (0x08, 3), (0x04, 3)
    ]
    
    byte_pool = [b for b, w in byte_distribution for _ in range(w)]
    
    with open(filename, "wb") as f:
        for _ in range(num_descriptors):
            descriptor = bytes(random.choices(byte_pool, k=descriptor_size))
            f.write(descriptor)

    print(f"Fake descriptor file '{filename}' created with {num_descriptors} descriptors.")

# generate_fake_desc_file



def generate_matches_file(feature_folder, frames_detected_dictionary, num_corners_total):
    print('Generating matches file...')

    key_list = sorted(frames_detected_dictionary.keys())

    file = open(f"{feature_folder}/matches.f.txt", "w")

    for _ in range(len(key_list)-1):
        current_key = key_list.pop(0)

        for second_key in key_list:

            file.write(f'{current_key} {second_key}\n')

            # Get the detection lists
            current_list = frames_detected_dictionary[current_key]
            second_list = frames_detected_dictionary[second_key]

            # Crop both arrays to the length of the shortest one
            min_length = min(len(current_list), len(second_list))
            current_cropped = current_list[:min_length]
            second_cropped = second_list[:min_length]

            # Compute number of matches
            result = np.logical_and(current_cropped, second_cropped)
            num_matches = np.sum(result) * num_corners_total
            file.write(f'{num_matches}\n')

            # Chech each row to see matches
            for i in range(min_length):
                if current_cropped[i] and second_cropped[i]:
                    # For each match, add all the corners to the matches list
                    for j in range(num_corners_total):
                        current_feature = (i*num_corners_total) + j
                        file.write(f'{current_feature} {current_feature}\n')

    file.close()
# generate_matches_file


if __name__ == "__main__":
    main()