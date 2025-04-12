import pandas as pd

def main():

    # Define the base path where the metrics files are located
    base_path = '/mnt/Marivi/IDS_sequences/ICCT2025/Experiments'
    
    camera_positions = ["CloseCameras", "FarCameras"]
    actions = ["standing", "walking"]
    resolutions = ["720", "1080"]
    methods = ["DA", "depthpro", "unidepth"]

    for camera in camera_positions:
        for action in actions:
            for resolution in resolutions:
                for method in methods:
                    metrics_file = f"{base_path}/{camera}_{action}_{resolution}_{method}/metrics/metrics.csv"

                    print(f'Loading metrics from {metrics_file}')

                    # Load the CSV file
                    df = pd.read_csv(metrics_file)

                    # Add loop variables as columns
                    df['CameraPosition'] = camera
                    df['Action'] = action
                    df['Resolution'] = resolution
                    df['Method'] = method

                    # Concatenate dataframes
                    if 'combined_df' in locals():
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
                    else:
                        combined_df = df

    # Save the combined dataframe to a new CSV file
    combined_metrics_file = 'Data/combined_metrics.csv'
    combined_df.to_csv(combined_metrics_file, index=False)
    print(f"Combined metrics saved to {combined_metrics_file}")
    

if __name__ == "__main__":
    main()
