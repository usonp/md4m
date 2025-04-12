import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

def main():
    metrics_file = 'Data/combined_metrics_union.csv'
    # Camera,CameraPosition,Action,Resolution
    group_by = 'Resolution'
    metrics_in_percentage = True

    # Load the CSV file
    df = pd.read_csv(metrics_file)

    # Convert numeric columns to float (in case they are read as strings)
    numeric_columns = ['PSNR', 'SSIM', 'LPIPS', 'Mask_IoU']
    percentage_columns = ['SSIM', 'Mask_IoU']
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Multiply by 100 the percentages
    if metrics_in_percentage:
        for col in percentage_columns:
            df[col] = df[col].apply(lambda x: x * 100)

    # Generate one dataframe per method, group them by group_by, and concatenate them
    methods = df['Method'].unique()
    grouped_dfs = []

    for method in methods:
        method_df = df[df['Method'] == method].groupby(group_by)[numeric_columns].agg(['mean', 'std']).reset_index()
        method_df[group_by] = method_df[group_by].apply(lambda x: f"{method}-{x}")
        grouped_dfs.append(method_df)

    # Concatenate all grouped dataframes
    concatenated_df = pd.concat(grouped_dfs, ignore_index=True)
    stats = concatenated_df.rename(columns={group_by: f'Method-{group_by}'})
    print(stats)

    # Export the stats dataframe as a LaTeX table
    latex_table = stats.to_latex(index=False, float_format="%.2f", caption="Metrics grouped by Method and Camera", label="tab:metrics_table")
    with open("metrics_table.tex", "w") as f:
        f.write(latex_table)

    # Repeat for plots
    grouped_dfs = []
    for method in methods:
        method_df = df[df['Method'] == method].copy()
        method_df[group_by] = method_df[group_by].apply(lambda x: f"{method}-{x}")
        grouped_dfs.append(method_df)

    # Concatenate all grouped dataframes
    concatenated_df = pd.concat(grouped_dfs, ignore_index=True)
    global_mean = concatenated_df[numeric_columns].mean()

    # Assign a unique color to each method
    method_colors = {method: color for method, color in zip(methods, sns.color_palette("tab10", len(methods)))}
    concatenated_df['Color'] = concatenated_df['Method'].map(method_colors)

    # --- BOX PLOTS ---
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        ax = sns.boxplot(
            x=group_by, 
            y=metric, 
            data=concatenated_df, 
            hue=group_by, 
            palette=concatenated_df.set_index(group_by)['Color'].to_dict(), 
            dodge=False
        )
        # ax = sns.boxplot(x=group_by, y=metric, data=concatenated_df)
        
        # Add a horizontal line for the global mean
        plt.axhline(global_mean[metric], color='red', linestyle='--', linewidth=2, label="Global Mean")
        
        # Add legend
        plt.legend()
        
        plt.title(f"Distribution of {metric}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # --- CONFIDENCE INTERVALS ---
    confidence_stats = concatenated_df.groupby(group_by)[numeric_columns].agg(['mean', confidence_interval]).reset_index()

    # Rename columns for clarity
    confidence_stats.columns = [group_by] + [f"{col[0]}_{col[1]}" for col in confidence_stats.columns[1:]]
    
    # Extract lower and upper bounds of confidence intervals
    for metric in numeric_columns:
        confidence_stats[f"{metric}_ci_lower"] = confidence_stats[f"{metric}_confidence_interval"].apply(lambda x: x[0])
        confidence_stats[f"{metric}_ci_upper"] = confidence_stats[f"{metric}_confidence_interval"].apply(lambda x: x[1])

    # Drop the tuple column
    confidence_stats = confidence_stats.drop(columns=[f"{metric}_confidence_interval" for metric in numeric_columns])

    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(numeric_columns, 1):
        plt.subplot(2, 2, i)
        
        # Scatter plot with error bars (confidence interval)
        plt.errorbar(
            confidence_stats[group_by], 
            confidence_stats[f"{metric}_mean"], 
            yerr=[confidence_stats[f"{metric}_mean"] - confidence_stats[f"{metric}_ci_lower"],
                confidence_stats[f"{metric}_ci_upper"] - confidence_stats[f"{metric}_mean"]],
            fmt='o', color='blue', ecolor='black', capsize=5, elinewidth=2, markersize=8
        )
        
        plt.title(f"{metric} Mean with 95% CI")
        plt.xlabel(group_by)
        plt.ylabel(metric)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Function to compute 95% confidence interval
def confidence_interval(series):
    return st.t.interval(0.95, len(series) - 1, loc=series.mean(), scale=st.sem(series))
    

if __name__ == "__main__":
    main()
