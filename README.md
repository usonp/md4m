# Monocular Depth Estimation for Multicamera Setups

This is the code implementation for the paper 📄 [Is Real-time Deep Learning-based Monocular Depth
Estimation accurate for Multi-Camera Setups?](https://doi.org/10.1109/icct-europe63283.2025.11157669)


[![Demo](assets/demo.gif)](assets/demo.mp4)

## Overview

This repository provides tools to perform the following tasks:

- (Optional) Intrinsic camera calibration and image distortion correction (see [monoDepth](monoDepth/README.md))
- Monocular depth estimation using deep learning models (see [monoDepth](monoDepth/README.md))
- Multi-camera calibration using a calibration pattern and OpenMVG (see [calibration](calibration/README.md))
- Adjusting the depth estimation to the multi-camera calibration for real-time execution (see [calibration](calibration/README.md))
- Computing objective image metrics using renderings obtained from the RGB+D content (see [metrics](metrics/README.md))

## Getting things ready

The software provided is written in Python, to install the required dependencies:

    pip install -r calibration/requirements.txt

Extra requirements to execute the Deep Learning models can be installed with:

    pip install -r monoDepth/extra_requirements.txt

Metrics also have specific requirements:

    pip install -r metrics/requirements.txt

Calibration tasks are handled by [OpenMVG](https://github.com/openMVG/openMVG.git). Follow the official instructions to compile the library and copy the resulting executables to `calibration/Software/OpenMVG`. Note that the software has only been tested on Ubuntu 22.

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

- `Checkerboard` contains captures of the [OpenCV checkerboard calibration pattern](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- `EncodedFiles/RGB` contains the images captured from the cameras used to compute depth
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

## ACKs
*This work was partially supported by projects PID2020-115132RB (SARAOS) and PID2023-148922OA-I00 (EEVOCATIONS) funded by MCIN/AEI/10.13039/501100011033 of the Spanish Government, HORIZON-IA-1010702-50 (XReco) funded by the European Union, TED2021-131690B-C31 (Revolution) funded by MCIN/AEI/10.13039/501100011033 and by NextGenerationEU/PRTR, and by the project UNICO-5G I+D TSI-063000-2021-80 (DISRADIO–Pilotos) funded by the Ministry of Digital Transformation of the Spanish Government and the NextGenerationEU (RRTP).*

Licensed under the MIT License.