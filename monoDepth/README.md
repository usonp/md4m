# Scripts for intrinsic calibration and depth computation

## Calibrating Intrinsic Parameters

With checkerboard images saved in `<Sequence path>/Checkerboard`, run:

    python intrinsic_calibration_opencv.py <Sequence path>

This will generate `<Sequence path>/Calibration/camera_intrinsics.json`.

## Perform Undistortion

Use the intrinsic calibration to undistort images:

    python undistort_images.py \
        <Sequence path>/Checkerboard \
        <Sequence path>/Checkerboard_undistorted \
        <Sequence path>/Calibration/camera_intrinsics.json

And for captures:

    python undistort_images.py \
        <Sequence path>/EncodedFiles/RGB \
        <Sequence path>/EncodedFiles/RGB_undistorted \
        <Sequence path>/Calibration/camera_intrinsics.json

Use these commands to rename the folders:

    mv <Sequence path>/Checkerboard <Sequence path>/Checkerboard_original
    mv <Sequence path>/Checkerboard_undistorted <Sequence path>/Checkerboard

    mv <Sequence path>/EncodedFiles/RGB <Sequence path>/EncodedFiles/RGB_original
    mv <Sequence path>/EncodedFiles/RGB_undistorted <Sequence path>/EncodedFiles/RGB

## Extract Calibration Files

To extract calibration files, monocular depth estimation will be applied to the checkerboard captures. Configure the parameters at the beginning of `calibration_sequence_processing.py`:

- `chessboardSize`: number of squares on your calibration pattern
- `num_cams`: number of cameras used
- `use_rectified_images`: `True` if undistortion was performed
- `depth_model`: model to be used - `DA2`/`DA2_metric`/`Depth_pro`/`UniDepth`
- `vit_encoder`: transformer backend to use (does not apply to `Depth_pro`) - `vits`/`vitb`/`vitl`

Again, with checkerboard images in `<Sequence path>/Checkerboard`, run:

    python calibration_sequence_processing.py <Sequence path>

The results will be a set of files inside `<Sequence path>/Calibration`. **Use these files with the [calibration](../calibration/README.md) software.**

## Monocular Depth Estimation+Correction

Once calibration is **complete** and successful, use the following software to extract depth information **for each camera**:

    python correct_depth_segmentation.py 

Modify the following parameters at the top of the Python script:

- `depth_model`: model to be used - `DA2`/`DA2_metric`/`Depth_pro`/`UniDepth`
- `vit_encoder`: transformer backend to use (does not apply to `Depth_pro`) - `vits`/`vitb`/`vitl`
- `use_rectified_images`: `True` if undistortion was performed
- `normalize_depth_images`: if set to `False`, depth will be stored as absolute measurements in mm; if set to `True`, depth will be stored as a relative value between `Z_near` and `Z_far`, both in mm
- `depth_correction`: `True` to apply the depth correction computed during calibration
- `using_masks`: `True` to mask the generated depth using the segmentation images from `<Sequence path>/EncodedFiles/Masks`
- `visualize_video`: show the results in real time

Some parameters can be provided via command line as follows:

    python correct_depth_segmentation.py \
        <Sequence path>/RGB/<cam_id> \
        <Sequence path>/Masks/<cam_id> \
        <Sequence path>/Calibration/Results/DepthTransformationParameters.yaml \
        <Sequence path>/Calibration/camera_intrinsics.json \
        <cam_id> \
        <Sequence path>/DEPTH/<cam_id> \
        <Sequence path>/DEPTH_masked/<cam_id> \

If masks are not available, set the `using_masks` parameter to `False` to ignore those inputs.

To run the software on all camera streams, use the following script:

    chmod +x correct_depth_segmentation.sh
    ./correct_depth_segmentation.sh