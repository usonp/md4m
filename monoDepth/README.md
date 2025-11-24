# Scripts for intrinsic calibration and depth computation

## Calibrating Intrinsic Parameters

With checkerboard captures on `<Sequence path>/Checkerboard`, run:

    python intrinsic_calibration_opencv.py <Sequence path>

This will generate the file `<Sequence path>/Calibration/camera_intrinsics.json`

## Perform Undistortion

The intrinsic calibration can be used to perform undistortion:

    python undistort_images.py \
        <Sequence path>/Checkerboard \
        <Sequence path>/Checkerboard_undistorted \
        <Sequence path>/Calibration/camera_intrinsics.json

And for captures:

    python undistort_images.py \
        <Sequence path>/EncodedFiles/RGB \
        <Sequence path>/EncodedFiles/RGB_undistorted \
        <Sequence path>/Calibration/camera_intrinsics.json

Use these command to rename the folders:

    mv <Sequence path>/Checkerboard <Sequence path>/Checkerboard_original
    mv <Sequence path>/Checkerboard_undistorted <Sequence path>/Checkerboard

    mv <Sequence path>/EncodedFiles/RGB <Sequence path>/EncodedFiles/RGB_original
    mv <Sequence path>/EncodedFiles/RGB_undistorted <Sequence path>/EncodedFiles/RGB


## Extract Calibration Files

To extract calibration files, monocular depth estimation will be applied to the checkboard captures, configure the parameters at the begining of the script `calibration_sequence_processing.py`:

- `chessboardSize`: number of squares on your calibration pattern
- `num_cams`: Number of cameras used
- `use_rectified_images`: `True` if undistortion was performed
- `depth_model`: model to be used - `DA2`/`DA2_metric`/`Depth_pro`/`UniDepth`
- `vit_encoder`: transformer backend to use (does not apply to `Depth_pro`) - `vits`/`vitb`/`vitl`

Again, with checkerboard captures on `<Sequence path>/Checkerboard`, run:

    python calibration_sequence_processing.py <Sequence path>

The results will be a set of files inside of the directory `<Sequence path>/Calibration`, **these are the files to be used with the [calibration](../calibration/README.md) software**.

## Monocular Depth Estimation+Correction

Once the calibration process is **done** (and it was successful), the following software can be used to extract the depth information **for each camera**:

    python correct_depth_segmentation.py 

The parameters should be modified on top of the Python script:

- `depth_model`: model to be used - `DA2`/`DA2_metric`/`Depth_pro`/`UniDepth`
- `vit_encoder`: transformer backend to use (does not apply to `Depth_pro`) - `vits`/`vitb`/`vitl`
- `use_rectified_images`: `True` if undistortion was performed
- `normalize_depth_images`: if set to `False`, depth will be stored as absolute measurements in mm; if set to `True`, depth will be stored as a relative value between `Z_near` and `Z_far`, both in mm
- `depth_correction`: `True` to apply the depth correction computer in the calibration
- `using_masks`: `True` to mask the generated depth using the segmentation images from `<Sequence path>/EncodedFiles/Masks`
- `visualize_video`: show the results in real time

Some parameters can be fed to the software by commandline as follows:

    python correct_depth_segmentation.py \
        <Sequence path>/RGB/<cam_id> \
        <Sequence path>/Masks/<cam_id> \
        <Sequence path>/Calibration/Results/DepthTransformationParameters.yaml \
        <Sequence path>/Calibration/camera_intrinsics.json \
        <cam_id> \
        <Sequence path>/DEPTH/<cam_id> \
        <Sequence path>/DEPTH_masked/<cam_id> \

Note that if masks are not available, the `using_masks` parameter can be set to `False` to ignore those inputs.

To run the software to all the camera streams the following script is provided:

    chmod +x correct_depth_segmentation.sh
    ./correct_depth_segmentation.sh