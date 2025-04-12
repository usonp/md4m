import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

def main():
    metrics_file = 'Data/combined_metrics_union.csv'
    # Camera,CameraPosition,Action,Resolution,Method
    group_by = ['CameraPosition', 'Action', 'Resolution']

    use_only_one_model = False
    # DA, depthpro, unidepth
    model = 'unidepth'

    metrics_in_percentage = True

    # Load the CSV file
    df = pd.read_csv(metrics_file)

    if use_only_one_model:
        # Filter the dataframe to only include the specified model
        df = df[df['Method'] == model]
        # Remove the 'Method' column
        df = df.drop(columns=['Method'])

    # Convert numeric columns to float (in case they are read as strings)
    numeric_columns = ['PSNR', 'SSIM', 'LPIPS', 'Mask_IoU']
    percentage_columns = ['SSIM', 'Mask_IoU']
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Multiply by 100 the percentages
    if metrics_in_percentage:
        for col in percentage_columns:
            df[col] = df[col].apply(lambda x: x * 100)


    # --- BOX PLOTS ---
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        for category in group_by:
            ax = sns.boxplot(x=category, y=metric, data=df)
            if metrics_in_percentage and metric in percentage_columns:
                plt.ylim(0, 100)
        plt.xlabel(None)
        # plt.xticks(rotation=45)
    # plt.legend()
    for ax in plt.gcf().axes:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(14)
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
