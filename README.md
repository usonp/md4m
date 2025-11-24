# Monocular Depth Estimation for Multicamera Setups

This is the code implementation for the paper 📄 [Is Real-time Deep Learning-based Monocular Depth
Estimation accurate for Multi-Camera Setups?](https://doi.org/10.1109/icct-europe63283.2025.11157669)


<div align="center">
    <video src="assets/demo.mp4" controls width="600">
        You should be seeing the video assets/demo.mp4.
    </video>
</div>

## Overview

This repository proveides tools to perform the following tasks:

- (Optional) Intrinsic camera calibration and image distorion correction (see [monoDepth](monoDepth/README.md))
- Monocular Depth estimation using Deep Learning Models (see [monoDepth](monoDepth/README.md))
- Multi-Camera calibration using a calibration pattern and OpenMVS (see [calibration](calibration/README.md))
- Adjusting the depth estimation to the multicamera calibration for real-time execution (see [calibration](calibration/README.md))
- Compute objective image metrics using renderings obtained from the RGB+D content (see [metrics](metrics/README.md))

## Getting things ready

The software provided is written in Python, to install the required dependencies:

    pip install -r calibration/requirements.txt

Extra requirements to execute the Deep Learning models can be installed with:

    pip install -r monoDepth/extra_requirements.txt

Metrics also have specific requirements:

    pip install -r metrics/requirements.txt

Calibration tasks are taken care by [OpenMVG](https://github.com/openMVG/openMVG.git), follow the official instructions to compile the library and copy the resulting executables to `calibration/Software/OpenMVG`. Note that the software has only been tested in Ubuntu 22.

To start processing a sequence with $N$ cameras, the following file structure is recommended:

    <Sequence path>
    ├── Checkerboard
    │   ├── 0
    │   |   ├── frame0.png
    │   |   ├── frame1.png
    │   |   ├── ...
    │   |   └── frameN.png
    │   ├── 1
    │   ├── 2
    │   ├── ...
    │   └── N
    └── EncodedFiles
        ├── RGB
        |   ├── 0
        |   ├── 1
        |   ├── 2
        |   ├── ...
        |   └── N
        └── Masks
            ├── 0
            ├── 1
            ├── 2
            ├── ...
            └── N

Where:

- `Checkerboard` are captures of the [OpenCV checkboard calibration pattern](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- `EncodedFiles/RGB` are the captures from the cameras to use to compute depth
- `EncodedFiles/Masks` is an optional folder with segmentation masks for each frame of the captures

## Citation

If you found this software useful, we would really appreciate it if you cite it in your work!

    @INPROCEEDINGS{11157669,
    author={Usón, Javier and Cabrera, Julián},
    booktitle={2025 IEEE International Conference on Consumer Technology-Europe (ICCT-Europe)}, 
    title={Is Real-time Deep Learning-based Monocular Depth Estimation accurate for Multi-Camera Setups?}, 
    year={2025},
    pages={1-5},
    doi={10.1109/ICCT-Europe63283.2025.11157669}}
