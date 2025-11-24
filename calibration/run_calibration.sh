#!/bin/bash

#####################################3
## Configure
CALIBRATION_FOLDER="Sequence/Calibration" # Path to calibration folder
INTRINSIC_JSON="$CALIBRATION_FOLDER/camera_intrinsics.json"

# 1920-1080/1280-720
WIDTH=1280
HEIGHT=720

# Detecting corners
num_corners_x=10
num_corners_y=5
square_size=101.6
# square_size=51

# Reconstruction
load_intrinsics=1
rectified_images=1
# GLOBAL/INCREMENTAL
MODE="INCREMENTAL"
group_camera_model="0"
f=$((HEIGHT * 5 / 4))

# FVV export
num_servers=1
#####################################3

#### Execution folders
# Input
DETECTEDPOINTS="$CALIBRATION_FOLDER/DetectedPoints"
DEPTHPOINTS="$CALIBRATION_FOLDER/DepthAtDetectedPoints"
PARAMETERS="$CALIBRATION_FOLDER/Parameters"

# Output
OUT_FOLDER="$CALIBRATION_FOLDER/Results"
# OUT_FOLDER="./Results"
IMAGE_FOLDER="$OUT_FOLDER/dummy_images"
FEAT_FOLDER="$OUT_FOLDER/corner_features"
PLOT_FOLDER="$OUT_FOLDER/calibration_plots"
SfM_FOLDER="$OUT_FOLDER/SfM"

#### Software folders
SW_FOLDER="./Software"
OMVG_FOLDER="$SW_FOLDER/OpenMVG"

# Remove previous reconstruction, create new folders
rm -r $OUT_FOLDER
mkdir -p $OUT_FOLDER
mkdir -p $IMAGE_FOLDER
mkdir -p $FEAT_FOLDER
mkdir -p $PLOT_FOLDER
mkdir -p $SfM_FOLDER


# Generate dummy images required for the SfM file
echo 'Generating Dummy Images'
python $SW_FOLDER/generate_dummy_images.py $DETECTEDPOINTS $IMAGE_FOLDER $WIDTH $HEIGHT

# Generate base SfM file 
echo "Executing Image Listing with f=$f"
"$OMVG_FOLDER/openMVG_main_SfMInit_ImageListing" -i $IMAGE_FOLDER -f $f -o $SfM_FOLDER -c 1 -g $group_camera_model

# Load intrinsics if required
if [ $load_intrinsics -eq 1 ]; then
    echo "Loading intrinsics"
    python $SW_FOLDER/add_intinsic_to_sfm.py $SfM_FOLDER/sfm_data.json $INTRINSIC_JSON $rectified_images
    # refine_intrinsic="ADJUST_ALL"
    refine_intrinsic="NONE"
else
    refine_intrinsic="ADJUST_ALL"
fi

# Create OpenMVG features (.feat), descriptors (.desc) and matches file from FVV calibration files
echo "Generating features and matches from calibration"
python $SW_FOLDER/generate_features_and_matches.py $DETECTEDPOINTS $FEAT_FOLDER $num_corners_x $num_corners_y $PLOT_FOLDER

# Compute Structure from Motion
echo 'Executing Strucuture from Motion'
"$SW_FOLDER/OpenMVG/openMVG_main_SfM" -i $SfM_FOLDER/sfm_data.json -m $FEAT_FOLDER -o $SfM_FOLDER -s $MODE -f $refine_intrinsic -M $MATCHES/matches.f.bin

# Calibration to JSON format
"$SW_FOLDER/OpenMVG/openMVG_main_ConvertSfM_DataFormat" -i $SfM_FOLDER/sfm_data.bin -o $SfM_FOLDER/calibrated.json

# Scale the calibration
echo "Scaling calibration with square size $square_size"
python $SW_FOLDER/scale_calibration.py $SfM_FOLDER/calibrated.json $PLOT_FOLDER $num_corners_x $num_corners_y $square_size

# Compute metrics for the calibration
echo "Computing metrics for the calibration"
python $SW_FOLDER/get_calibration_stats.py $SfM_FOLDER/calibrated_scaled.json $OUT_FOLDER $PLOT_FOLDER

# View scaled reconstruction
echo "Preparing interactive visualization of calibration"
python $SW_FOLDER/plot_calibration_3D.py $SfM_FOLDER/calibrated_scaled.json

# Export calibration as .yaml for FVV
echo "Exporting calibration as $OUT_FOLDER/synthesisConfiguration.yaml"
python $SW_FOLDER/oMVG2FVVLive.py $SfM_FOLDER/calibrated_scaled.json $DEPTHPOINTS $num_servers $OUT_FOLDER/synthesisConfiguration.yaml

# Perform depth correction
echo "Performing depth correction"
python $SW_FOLDER/correct_depth.py $SfM_FOLDER/calibrated_scaled.json $DEPTHPOINTS $PARAMETERS $num_corners_x $num_corners_y $OUT_FOLDER $PLOT_FOLDER

# Optional: Check depth correction
# echo "Checking depth correction"
# python $SW_FOLDER/check_depth_correction.py $SfM_FOLDER/calibrated_scaled.json $DETECTEDPOINTS $DEPTHPOINTS $num_corners_x $num_corners_y

echo "Calibration finished!"