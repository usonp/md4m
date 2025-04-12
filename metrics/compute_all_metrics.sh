# This script computes all the metrics for the different configurations of the experiments.

BASE_PATH="/mnt/Marivi/IDS_sequences/ICCT2025/Experiments"

# Define arrays for experiments
camera_positions=("CloseCameras" "FarCameras")
actions=("standing" "walking")
resolutions=("720" "1080")
methods=("DA" "depthpro" "unidepth")

# Loop through all combinations and execute the tests
for resolution in "${resolutions[@]}"; do 
    for action in "${actions[@]}"; do
        for camera in "${camera_positions[@]}"; do
            for method in "${methods[@]}"; do
                CURRENT_PATH="${BASE_PATH}/${camera}_${action}_${resolution}_${method}"
                echo "Running metrics for ${camera}_${action}_${resolution}_${method}"
                mkdir -p "$CURRENT_PATH/metrics_union"
                python Compute_metrics.py "$CURRENT_PATH/EncodedFiles" "$CURRENT_PATH/metrics_union"
            done
        done
    done
done