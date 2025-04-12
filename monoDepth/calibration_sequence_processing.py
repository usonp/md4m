#!/usr/bin/python3

import glob
import sys
from os import makedirs
from os.path import join

import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_model, compute_depth, bilinear_interpolation
from utils import intrinsic_dictionary_from_file, coolPrint


def main():

	################################
	# Parameters
	chessboardSize = (10, 5)
	num_cams = 4
	use_rectified_images = False

	# Depth models: DA2/DA2_metric/Depth_pro/UniDepth
	depth_model = 'DA2_metric'
	# encoders vits/vitb/vitl -- vitg
	vit_encoder = 'vitl'
	
	input_path = "sequence"
	frame_name = "frame_"

	showImages = False
	waitKeyValue = 33
	megaVerbose = False
	################################

	# Read path from command line or use the default one
	if len(sys.argv) > 1:
		input_path = str(sys.argv[1])

	# Create directories
	outPath = join(input_path, 'Calibration')
	depthPath = join(outPath, 'DepthAtDetectedPoints')
	detectedPath = join(outPath, 'DetectedPoints')
	paramsPath = join(outPath, 'Parameters')
	depthFpath = join(depthPath, 'Depth')
	pointFpath = join(detectedPath, 'CalibCam')
	paramsFPath = join(paramsPath, 'Parameters')
	IntrisicsPath = join(outPath, 'camera_intrinsics.json')

	makedirs(depthPath, exist_ok=True)
	makedirs(detectedPath, exist_ok=True)
	makedirs(paramsPath, exist_ok=True)


	# Load intrinsics and create a dictionary
	IntrisicDict = intrinsic_dictionary_from_file(IntrisicsPath, use_rectified_images)

	# Load depth anything model
	model, transform = load_model(depth_model, encoder=vit_encoder)

	## Find checkboard corners and create the corresponding calibration files
	# termination criteria
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
	objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

	# Arrays to store object points and image points from all the images.
	# objpoints = []  # 3d point in real world space
	imgpoints = [None]*num_cams  # 2d points in image plane.

	first_frame_number_list = [0]*num_cams
	last_frame_number_list = [0]*num_cams
	depth_file_vector = [None]*num_cams
	point_file_vector = [None]*num_cams

	for i in range(num_cams):
		# Get the first and last frame number in the folder
		image_files = glob.glob(join(input_path, 'Checkerboard/{}/*.png'.format(i)))
		frame_number_list = [int(f.split('/')[-1].split('.')[0].replace(frame_name, '')) for f in image_files]
		first_frame_number_list[i] = min(frame_number_list)
		last_frame_number_list[i] = max(frame_number_list)
		depth_file_vector[i] = open(depthFpath + str(i).zfill(2) + '.txt', "w")
		point_file_vector[i] = open(pointFpath + str(i).zfill(2) + '.txt', "w")
		imgpoints[i] = []

		# Generate dummy parameter files
		with open(paramsFPath + f"{i}", "w") as f:
			f.write("DEVICE: INTEL\n\n")
			f.write(f"SERIAL NUMBER: {i}\n\n")
			f.write("Node: 0\n")
			f.write(f"Stream: {i}\n")
			f.write(f"Camera: {i}\n\n")
			f.write("zNear: 1000\n")
			f.write("zFar: 6000\n")

	# Select the range of values to process
	first_frame_number = min(first_frame_number_list)
	last_frame_number = max(last_frame_number_list)
	print(f"First frame number: {first_frame_number}")
	print(f"Last frame number: {last_frame_number}")
	success_count = 0
	failed_count = 0
	coolPrint('Processing frames...')
	for i in tqdm(range(first_frame_number, last_frame_number+1), colour='cyan'):

		rets = [None]*num_cams
		corners = [None]*num_cams

		for j in range(num_cams):

			frame_path = join(input_path, f'Checkerboard/{j}/{frame_name}{i}.png')

			# Read color frames - skip them if not available
			img = cv.imread(frame_path)
			if img is None:
				coolPrint(f"Failed to read image {frame_path}", color='red')
				continue
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

			# Obtain corners
			rets, corners = cv.findChessboardCorners(gray, chessboardSize, None)

			# If found, add object points, image points (after refining them)
			if rets:
	    		# add object points, image points (after refining them)
				cornersSub = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
				imgpoints[j].append(cornersSub)

				# compute depth frames
				depth = compute_depth(img, depth_model, model, IntrisicDict[j], transform=transform)

				depth_file_vector[j].write('{} {}\n'.format(j,i))
				point_file_vector[j].write('{} {}\n'.format(j,i))

				for point in cornersSub:
					x, y = point[0][0],point[0][1]
					point_file_vector[j].write('{}, {}\n'.format(x, y))
					interpolated_depth = bilinear_interpolation(depth, x, y)
					if interpolated_depth < 1e-6 or interpolated_depth > 1e6:
						print(f'Invalid depth in coordenate ({y},{x}) cam {j} frame{i}')
						depthValue = 0
					else:
						depthValue = interpolated_depth
					depth_file_vector[j].write(str(depthValue) + '\n')
					if megaVerbose:
						print(f'x: {x}, y: {y} -> Z: {depthValue}')

				# Draw and display the corners
				if showImages:
					cv.drawChessboardCorners(img, chessboardSize, cornersSub, rets)
					Z_norm = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
					Z_colormap = cv.applyColorMap(Z_norm, cv.COLORMAP_INFERNO)
					cv.drawChessboardCorners(Z_colormap, chessboardSize, cornersSub, rets)
					cv.imshow('img {}'.format(j+1), img)
					cv.imshow('Depth {}'.format(j+1), Z_colormap)

				if megaVerbose:
					plt.imshow(depth, cmap='inferno', norm=plt.Normalize(vmin=np.min(depth), vmax=np.max(depth)))
					plt.scatter(cornersSub[:,:,0], cornersSub[:,:,1], color='black', marker='x', label='Points')
					plt.show()

				success_count += 1
			else:
				failed_count += 1
				# print('Did not find corners, count {}'.format(failed_count))

		if showImages:
			cv.waitKey(waitKeyValue)
		

	# Process finished
	coolPrint(f'Dropped frames: {failed_count} ({failed_count/(failed_count+success_count)*100:.2f}%)')

	for f in depth_file_vector:
		f.close()
	for f in point_file_vector:
		f.close()
	
	if showImages:
		cv.destroyAllWindows()

# END main


if __name__ == '__main__':
    main()