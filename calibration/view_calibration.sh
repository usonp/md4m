#!/bin/bash

#####################################3
## Configure
CALIBRATION_FOLDER="/mnt/Marivi/IDS_sequences/ICCT2025/Experiments/FarCameras_standing_720_unidepth/Calibration"
OUT_FOLDER="$CALIBRATION_FOLDER/Results"


SW_FOLDER="./Software"
SfM_FOLDER="$OUT_FOLDER/SfM"

echo "Preparing interactive visualization of calibration"
python $SW_FOLDER/plot_calibration_3D.py $SfM_FOLDER/calibrated_scaled.json
