# Scripts for depth computation

## Calibrating Intrinsic Parameters

With checkerboard captures on `<Base content path>/Checkerboard`, run:

    python intrinsic_calibration_opencv.py <Base content path>

This will generate the file `<Base content path>/Calibration/camera_intrinsics.json`

## Perform Undistortion

The intrinsic calibration can be used to perform undistortion:

    python undistort_images.py <Base content path>/Checkerboard <Base content path>/Checkerboard_undistorted <Base content path>/Calibration/camera_intrinsics.json

And for captures:

    python undistort_images.py \
        <Base content path>/EncodedFiles/RGB \
        <Base content path>/EncodedFiles/RGB_undistorted \
        <Base content path>/Calibration/camera_intrinsics.json

Remember to switch the names of the folders afterwards to avoid problems

## Extract Calibration Files

Again, with checkerboard captures on `<Base content path>/Checkerboard`, run:

    python calibration_sequence_processing.py <Base content path>

The results will be a set of files inside of the directory `<Base content path>/Calibration`, **these are the files to be used with the calibration software**.

## Monocular Depth Estimation+Correction

Once the calibration process is done (and it was successful), the following software can be used to extract the depth information for each camera:

    python correctDepthSegmentation.py 

The parameters should be modified in the Python script, or they can be fed to the software by commandline as follows:

    python correctDepthSegmentation.py \
        <Base content path>/RGB/<cam_id> \
        <Base content path>/Masks/<cam_id> \
        <Base content path>/Calibration/Results/DepthTransformationParameters.yaml \
        <Base content path>/Calibration/camera_intrinsics.json \
        <cam_id> \
        <Base content path>/DEPTH/<cam_id> \
        <Base content path>/DEPTH_masked/<cam_id> \

Note that if masks are not available, the `using_masks` parameter can be set to `False` to ignore those inputs.