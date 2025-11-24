# Metrics

This folder contains the scripts used to compute the metrics and plot the figures presented in the paper.

## Installation

The requirements can be installed in a Conda environment as follows:

    conda create -n md4m_metrics python==3.10
    conda activate md4m_metrics
    pip install -r requirements.txt

## Computing metrics

The `Compute_metrics.py` script expects the following folder structure:

    <Sequence path>
    └── EncodedFiles
        ├── RGB
        │   ├── 0
        │   │   ├── frame0.png
        │   │   ├── frame1.png
        │   │   ├── ...
        │   │   └── frameN.png
        │   ├── 1
        │   ├── 2
        │   ├── ...
        │   └── N
        └── RENDERED
            ├── 0
            ├── 1
            ├── 2
            ├── ...
            └── N

`RGB` contains the original captures, and `RENDERED` contains the virtual views to be compared for each of the $N$ cameras.

The complete results presented in the paper can be analyzed using the data in the `Data` folder.