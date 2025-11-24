#################################
# Software
DeepDepth="correct_depth_segmentation.py"
PYTHON="python"

#############################################
# PARAMETERS:
SEQUENCE_PATH="/mnt/DatosTeraWatos/IDS_sequences/test_720"
SEQUENCE="$SEQUENCE_PATH/EncodedFiles"
IN_calibration="$SEQUENCE_PATH/Calibration/Results/DepthTransformationParameters.yaml"
IN_intrinsics="$SEQUENCE_PATH/Calibration/camera_intrinsics.json"

IN_color="$SEQUENCE/RGB"
IN_masks="$SEQUENCE/SAM_masks"

OUT_frames="$SEQUENCE/DEPTH"
OUT_frames_seg="$SEQUENCE/DEPTH_masked"

NUM_CAMERAS=4

#############################################

echo $IN_color

echo "Computing depth..."
for ((i = 0; i < NUM_CAMERAS; i++));
do
    echo "  Processing view $i"
    mkdir -p $OUT_frames/$i 
    mkdir -p $OUT_frames_seg/$i 
    $PYTHON $DeepDepth $IN_color/$i $IN_masks/$i $IN_calibration $IN_intrinsics $i $OUT_frames/$i $OUT_frames_seg/$i
done