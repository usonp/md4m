import os
import sys
import glob
import yaml

from tqdm import tqdm
from natsort import natsorted

# for image processing
import cv2
import numpy as np

from utils import load_model, compute_depth, intrinsic_dictionary_from_file
from utils import fix_depth_linear, normalize_depth_16bits, coolPrint, show_depth


def main():

    #################################
    # To Configure:

    # Depth models: DA2/DA2_metric/Depth_pro/UniDepth
    depth_model = 'UniDepth'
    # encoders vits/vitb/vitl -- vitg
    vit_encoder = 'vitl'

    use_rectified_images = False
    normalize_depth_images = False
    Z_near = 500
    Z_far = 4000
    depth_correction = True
    using_masks = True

    visualize_video = False
    wait = 1 # opencv waitkey value, 0 to stop at each frame
    
    #################################

    # Default paths, can be provided by command line
    base_path = 'sequence'
    inPath_C = f'{base_path}/RGB/2'
    inPath_M = f'{base_path}/SAM_masks/2'
    
    regression_file = '/mnt/Marivi/IDS_sequences/Small_calib_pattern_720/Calibration/Results/DepthTransformationParameters.yaml'
    intrinsic_file = '/mnt/Marivi/IDS_sequences/Small_calib_pattern_720/Calibration/camera_intrinsics.json'
    camera_id = 2
    
    outPath = f'{base_path}/DEPTH/2'
    outPath_mask = f'{base_path}/DEPTH_mask/2'

    #################################


    # Read from command line if user provided arguments
    if len(sys.argv) > 7:
        inPath_C = str(sys.argv[1])
        inPath_M = str(sys.argv[2])
        regression_file = str(sys.argv[3])
        intrinsic_file = str(sys.argv[4])
        camera_id = int(sys.argv[5])
        outPath = str(sys.argv[6])
        outPath_mask = str(sys.argv[7])
        print('Parameters read from command line')
    else:
        print('Not enough arguments given, using default parameters')


    # Get all inputs
    color_paths = natsorted(glob.glob(os.path.join(inPath_C, '*.png')), reverse=False)
    masks_paths = natsorted(glob.glob(os.path.join(inPath_M, '*.png')), reverse=False)
    img_names = [os.path.basename(x) for x in color_paths]
    IntrisicDict = intrinsic_dictionary_from_file(intrinsic_file, use_rectified_images)

    color_len = len(color_paths)
    masks_len = len(masks_paths)
    print(f'sizes - color: {color_len} masks: {masks_len}')
    if color_len != masks_len:
        coolPrint('[ERROR]: DIFFERENT NUMBER OF FRAMES ): check what happened...', 'red')
        exit(1)

    # Load pretrained model
    model, transform = load_model(depth_model, encoder=vit_encoder)

    # Create output folders
    os.makedirs(outPath, exist_ok=True)
    if using_masks:
        os.makedirs(outPath_mask, exist_ok=True)

    # Read calibration data
    if depth_correction:
        with open(regression_file, 'r') as yaml_file:
            regression_data = yaml.safe_load(yaml_file)[str(camera_id)]

    # To visualize as a video
    if visualize_video:
        cv2.namedWindow('Color', cv2.WINDOW_NORMAL)  
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL) 
        cv2.resizeWindow('Color', 1280, 720)
        cv2.resizeWindow('Depth', 1280, 720) 


    coolPrint('Processing images...')
    # Here we start iterating over every image
    for color_p, mask_p, file_name in tqdm(zip(color_paths, masks_paths, img_names), total=len(color_paths), colour='cyan'):

        # Read image and mask
        color = cv2.imread(color_p)
        mask  = cv2.imread(mask_p, -1) > 128 # make mask binary

        # Infere and correct with calibration
        output = compute_depth(color, depth_model, model, IntrisicDict[camera_id], transform=transform)
        if depth_correction:
            corrected_output_float = fix_depth_linear(output, regression_data)
        else:
            corrected_output_float = output * 1000 # m to mm

        # Normalize (optional) and make 16 bits
        if normalize_depth_images:
            corrected_output = normalize_depth_16bits(corrected_output_float, Z_near, Z_far)
        else:
            corrected_output = corrected_output_float.astype(np.uint16)

        # Remove background from result
        if using_masks:
            corrected_output_masked = corrected_output.copy()
            corrected_output_masked[np.logical_not(mask)] = 50000

        # Save results
        cv2.imwrite(os.path.join(outPath,file_name), corrected_output)
        if using_masks:
            cv2.imwrite(os.path.join(outPath_mask,file_name), corrected_output_masked)

        # View results
        if visualize_video:
            cv2.imshow('Color', color)
            # show_depth('Depth', corrected_output_masked)
            show_depth('Depth', corrected_output, Z_near=Z_near, Z_far=Z_far)
            key = cv2.waitKey(wait)
            if key == 27: # pres Esc to exit
                coolPrint('ABORTING', 'yellow')
                break
                
    # Pro-programming
    else:
        coolPrint(f"Processing completed for {inPath_C}")

# End main

if __name__ == '__main__':
   main()
