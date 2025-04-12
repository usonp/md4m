import glob
import sys
import json
from os import makedirs
from os.path import join

import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import coolPrint


def main():
    ################################
    # Parameters
    num_cams = 4
    input_path = "sequence/Checkerboard"
    output_path = "sequence/Checkerboard_undistorted"
    calib_file = "sequence/Calibration/camera_intrinsics.json"
    show_images = False
    ################################
    
    # Get paths from commandline if present
    if len(sys.argv) > 3:
        input_path = str(sys.argv[1])
        output_path = str(sys.argv[2])
        calib_file = str(sys.argv[3])

    coolPrint("Loading calibration data...", "blue")
    calibration = load_calibration(calib_file)

    # Undistort all images
    for cam_id in range(num_cams):
        image_paths = glob.glob(join(input_path, f'{cam_id}/*.png'))
        if image_paths:
            undistort_images(cam_id, image_paths, calibration, output_path, show_images)

    coolPrint("Undistortion process completed!", "green")
    if show_images:
        cv.destroyAllWindows()


def load_calibration(calib_file):
    """Load camera calibration data from JSON file."""
    with open(calib_file, 'r') as f:
        return json.load(f)


def undistort_images(cam_id, image_paths, calibration, output_path, show_images=True):
    """Undistort images using the calibration parameters and save them."""
    makedirs(join(output_path, str(cam_id)), exist_ok=True)

    # mtx = np.array(calibration[str(cam_id)]['camera_matrix'])
    # dist = np.array(calibration[str(cam_id)]['dist_coeff'])
    # new_camera_mtx = np.array(calibration[str(cam_id)]['new_camera_matrix'])
    mapx = np.array(calibration[str(cam_id)]['rectification_map_x'], dtype=np.float32)
    mapy = np.array(calibration[str(cam_id)]['rectification_map_y'], dtype=np.float32)

    for image_path in tqdm(image_paths, desc=f'Undistorting Camera {cam_id}', colour='blue'):
        img = cv.imread(image_path)
        undistorted_img = cv.remap(img, mapx, mapy, interpolation=cv.INTER_LINEAR)

        output_file = join(output_path, str(cam_id), image_path.split('/')[-1])
        cv.imwrite(output_file, undistorted_img)

        if show_images:
            cv.imshow(f'Original Camera {cam_id}', img)
            cv.imshow(f'Undistorted Camera {cam_id}', undistorted_img)
            key = cv.waitKey(0)
            if key == 27:  # Press ESC to exit early
                cv.destroyAllWindows()
                return

if __name__ == '__main__':
    main()
