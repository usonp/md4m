import glob
import sys
import json
import os
from os.path import join

import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import coolPrint

def main():
    
	################################
	# Parameters
    chessboard_size = (10, 5)
    num_cams = 4
    alpha = 0
    base_path = "sequence"
	################################

    # Get base path from commandline if pressent
    if len(sys.argv) > 1:
        base_path = str(sys.argv[1])

    input_path = f"{base_path}/Checkerboard"
    output_folder = f"{base_path}/Calibration"
    output_file = f"{output_folder}/camera_intrinsics.json"

    os.makedirs(output_folder, exist_ok=True)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    camera_intrinsics = {}
    
	# Compute intrinsic for each camera
    for cam_id in range(num_cams):
        image_paths = glob.glob(join(input_path, f'{cam_id}/*.png'))
        imgpoints, objpoints, image_shape = process_camera_images(cam_id, image_paths, chessboard_size, criteria)
        
        if imgpoints and objpoints:
            camera_intrinsics[cam_id] = compute_intrinsics(imgpoints, objpoints, image_shape, alpha)
    
    with open(output_file, 'w') as f:
        json.dump(camera_intrinsics, f, indent=4)
    
    coolPrint(f"Calibration completed and saved to {output_file}")
    

def compute_intrinsics(imgpoints, objpoints, image_shape, alpha):
    """Compute camera intrinsics using collected image and object points."""
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )
    
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, image_shape, alpha, image_shape)
    
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, image_shape, cv.CV_32FC1)
    
    return {
        'camera_matrix': mtx.tolist(),
        'dist_coeff': dist.tolist(),
        'new_camera_matrix': new_camera_mtx.tolist(),
        'rectification_map_x': mapx.tolist(),
        'rectification_map_y': mapy.tolist()
    }


def process_camera_images(camera_id, image_paths, chessboard_size, criteria):
    """Process images from a single camera and return detected points."""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    imgpoints = []
    objpoints = []
    
    for image_path in tqdm(image_paths, desc=f'Processing Camera {camera_id}', colour='cyan'):
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            corners_sub = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_sub)
            objpoints.append(objp)
        else:
            coolPrint(f'Corners not found on image: {image_path}', color='yellow')
    
    return imgpoints, objpoints, gray.shape[::-1] if imgpoints else None


if __name__ == '__main__':
    main()
