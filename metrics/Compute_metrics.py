import os
import csv
from math import log10, sqrt

import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import torch
import torchvision.transforms as transforms
import lpips
import sys

# LPIPS stuff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
# Load and preprocess two images
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] as expected by LPIPS
])


def main():
    base_path = '/mnt/Marivi/IDS_sequences/UsonWalking_1080/EncodedFiles'
    metrics_file = 'metrics.csv'
    metrics_report = 'report.txt'
    use_masks = True
    mask_folder = "SAM_masks"

    if len(sys.argv) >= 3:
        print("Reading parameters from command line")
        base_path = sys.argv[1]
        output_path = sys.argv[2]

        metrics_file = os.path.join(output_path, 'metrics.csv')
        metrics_report = os.path.join(output_path, 'report.txt')

    # Create the CSV file and add a header
    csvfile = open(metrics_file, 'w', newline='')
    csvwriter = csv.writer(csvfile)
    columns = ['Camera', 'Frame', 'MSE', 'PSNR', 'SSIM', 'LPIPS']
    if use_masks:
        columns.append('Mask_IoU')
    csvwriter.writerow(columns)
    
    report_file = open(metrics_report, 'w')

    rendered_base_path = os.path.join(base_path, 'RENDERED')
    rgb_base_path = os.path.join(base_path, 'RGB')
    mask_base_path = os.path.join(base_path, mask_folder)


    for folder_name in os.listdir(rendered_base_path):
        rendered_path = os.path.join(rendered_base_path, folder_name)
        rgb_path = os.path.join(rgb_base_path, folder_name)
        mask_path = os.path.join(mask_base_path, folder_name)
        
        mse_list = []
        psnr_list = []
        ssim_list = []
        lpips_list = []
        iou_list = []

        if os.path.isdir(rendered_path) and os.path.isdir(rgb_path):
            for frame_name in tqdm(os.listdir(rendered_path), desc=f"Camera {folder_name}"):
                image_path1 = os.path.join(rgb_path, frame_name)
                image_path2 = os.path.join(rendered_path, frame_name)

                if os.path.isfile(image_path1) and os.path.isfile(image_path2):
                    image1 = load_image(image_path1)
                    image2 = load_image(image_path2)

                    # Load masks
                    mask = None
                    if use_masks:
                        mask_path1 = os.path.join(mask_path, frame_name)
                        if os.path.isfile(mask_path1):
                            mask1 = obtain_original_mask(mask_path1)
                            mask2 = obtain_rendered_mask(image2)

                            mask_intersection = np.logical_and(mask1, mask2)
                            mask_union = np.logical_or(mask1, mask2)
                            sum_intersection = np.sum(mask_intersection)
                            sum_union = np.sum(mask_union)
                            if sum_intersection == 0 or sum_union == 0:
                                continue # The masks are empty
                            mask_iou = sum_intersection / sum_union

                            # Final mask is intersection/union 
                            mask = mask_union
                        else:
                            print(f'[Warning]: mask file missing: {mask_path1}')
                            mask_iou = 0
                        iou_list.append(mask_iou)   

                    mse = compute_mse(image1, image2, mask=mask)
                    psnr = compute_psnr(mse)
                    ssim = compute_ssim(image1, image2, mask=mask)
                    lpips = compute_lpips(image1, image2, mask=mask)

                    # Store results
                    mse_list.append(mse)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    lpips_list.append(lpips)
                    row = [folder_name, frame_name, mse, psnr, ssim, lpips]
                    if use_masks:
                        row.append(np.mean(mask_iou))
                    csvwriter.writerow(row)

            # Compute average metrics
            avg_mse = np.mean(mse_list)
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            avg_lpips = np.mean(lpips_list)
            
            report_message = f"Camera {folder_name}: MSE: {avg_mse} - PSNR: {avg_psnr} dB - SSIM: {avg_ssim} - LPIPS: {avg_lpips}"
            if use_masks:
                avg_iou = np.mean(iou_list)
                report_message += f" - Mask IoU: {avg_iou}"
            print(report_message)
            report_file.write(f"{report_message}\n")

    # Close the files
    csvfile.close()
    report_file.close()


def load_image(image_path):
    return cv2.imread(image_path)

def compute_mse(image1, image2, mask = None):
    image1 = image1.astype(np.float32, copy=False)
    image2 = image2.astype(np.float32, copy=False)
    sqr_diff = (image1 - image2) ** 2
    if mask is not None:
        sqr_diff = sqr_diff[mask]
    return np.mean(sqr_diff)

def compute_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 20 * log10(max_pixel / sqrt(mse))

def compute_ssim(image1, image2, mask=None):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ssim_val, ssim_image = ssim(gray_image1, gray_image2, full=True)
    if mask is not None:
        ssim_image = ssim_image[mask]
        return np.mean(ssim_image)
    else:
        return ssim_val

def compute_lpips(image1, image2, mask=None):
    img1 = transform(image1).unsqueeze(0).to(device)  # Add batch dimension
    img2 = transform(image2).unsqueeze(0).to(device) 

    if mask is not None:
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device)
        img1 = img1 * mask
        img2 = img2 * mask

    # Compute LPIPS distance
    return loss_fn_alex(img1, img2).item()

def obtain_original_mask(mask_path, threshold=128):
    return load_image(mask_path)[:,:,0] > threshold

# Default is saturated green
def obtain_rendered_mask(rendered, value=(0,255,0)):
    return np.logical_not(rendered[:,:] == value)[:,:,0]


if __name__ == "__main__":
    main()
