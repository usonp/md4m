# OpenMVG Calibration + Depth correction
The software here processes the results from `../monoDepth/calibration_sequence_processing.py` to obtain:

- Intrinsic + Extrinsic camera calibration using [OpenMVG](https://github.com/openMVG/openMVG.git)
- Depth correction parameters to adjust monocular depth estimation based on the multicamera calibration

## Installation
The requirements can be installed in a Conda environment as follows:

    conda create -n md4m_calibration python==3.11
    conda activate md4m_calibration
    pip install -r requirements.txt

The library [OpenMVG](Software/OpenMVG/README.md) is required to calibrate cameras.

## Calibration

Configure the parameters in the script `run_calibration.sh`:

- `WIDTH` and `HEIGHT`: resolution of the captures
- `num_corners_x` and `num_corners_y`: number of corners in the calibration pattern (number of squares - 1)
- `square_size`: size of the squares of the calibration pattern in mm
- `load_intrinsics`: set to 1 to load intrinsics from `<Sequence path>/Calibration/camera_intrinsics.json`, set to 0 to let OpenMVG compute intrinsics from scratch
- `rectified_images`: set to 1 if undistortion was performed, set it to 0 otherwise
- `MODE`: OpenMVG calibration algorithm used (`GLOBAL`/`INCREMENTAL`)
- `group_camera_model`: set to 0, only set to 1 if all the images from all streams come from the same camera.
- `num_servers`: this parameter is only relevant for the FVV Live calibration file 

To run the software:

    chmod +x run_calibration.sh
    ./run_calibration.sh

Once the calibration is done, the following script can be used to visualize the calibration in a 3D plot:

    chmod +x view_calibration.sh
    ./view_calibration.sh

## Results and plots

The script creates a folder `<Sequence path>/Calibration/Results` containing all the output files:

- `calibration_plots`: Folder containing figures to evaluate the quality of the calibration:
    - `detected_points`: figures with the combination of all the corners extracted for each camera
    - `reprojection_error`: figures with all triangulated features reprojected into camera space plotted on top of detected points. Additionally, histograms and boxplots of the reprojection error measured in pixels
    - `scale`: Histograms and plots of the distances measured to scale the calibration into mm
    - `depth_correction`: Plots of the quadratic line fitted to adjust depth estimation to the multicamera calibration
- `SfM/calibrated_scaled.json`: OpenMVG file containing the calibration scaled to mm
- `synthesisConfiguration.yaml`: intrinsic and extrinsic calibration in FVV Live format (summary of the OpenMVG file)
- `calibration_stats.json`: stats for the reprojection error
- `DepthTransformationParameters.yaml`: Regression parameters to be used with `../monoDepth/correctDepthSegmentation.py`

The rest of files are intermediate results and can be ignored.
