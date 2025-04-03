import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import ast
import os

# Create plots directory in the evaluation folder
base_dir = Path('/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/evaluation')
plot_dir = base_dir / 'plots'
plot_dir.mkdir(exist_ok=True)
print(f"Created/verified plots directory at: {plot_dir.absolute()}")

# Read the data
file_path = Path('/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/results/pipeline_runs/pipeline_with_all_models_final_20250403_122938/validation_with_transformer_extraction.csv')
df = pd.read_csv(file_path)

sections = ['cleaned', 'decision', 'analysis', 'reasons', 'conclusion']
models = ['basic', 'bert_anomaly', 'roberta_balanced']

# Get maximum ranges for consistent scaling
def get_section_max_values(df):
    max_values = {}
    for section in sections:
        section_max = 0
        for model in models:
            col_name = f"{section}_{model}_extraction_count"
            if col_name in df.columns:
                # Use 99.5th percentile to avoid extreme outliers affecting the scale
                val = df[col_name].quantile(0.995)
                section_max = max(section_max, val)
        # Round up to nearest multiple of 5 or 10 for clean axes
        if section_max > 100:
            section_max = np.ceil(section_max / 10) * 10
        else:
            section_max = np.ceil(section_max / 5) * 5
        max_values[section] = section_max
    return max_values

max_values = get_section_max_values(df)
print("Maximum values for scaling by section:", max_values)

# Find the max frequency scale for each section
def get_frequency_max(df, section, bin_width=5):
    freqs = {}
    for model in models:
        col_name = f"{section}_{model}_extraction_count"
        if col_name in df.columns:
            # Get header column if it exists
            header_col = f"has_{section}_headers"
            if header_col in df.columns:
                data = df[df[header_col] == 1][col_name]
            else:
                data = df[col_name]
                
            # Create fixed-width bins
            bins = np.arange(0, max_values[section] + bin_width, bin_width)
            hist, _ = np.histogram(data, bins=bins)
            freqs[model] = max(hist)
    
    # Return maximum frequency + 10% padding
    return max(freqs.values()) * 1.1 if freqs else 100

# 1. Histograms for each section - ALL MODELS with fixed bins
for section in sections:
    plt.figure(figsize=(12, 8))
    
    # Fixed bin width for consistent comparison
    bin_width = 5 if max_values[section] <= 50 else 10
    bins = np.arange(0, max_values[section] + bin_width, bin_width)
    
    # Get the maximum frequency for y-axis scaling
    max_freq = get_frequency_max(df, section, bin_width)
    
    for model in models:
        col_name = f"{section}_{model}_extraction_count"
        header_col = f"has_{section}_headers"
        
        if col_name in df.columns:
            if header_col in df.columns:
                section_df = df[df[header_col] == 1]
            else:
                section_df = df
            
            plt.hist(section_df[col_name], bins=bins, alpha=0.5, label=model)
    
    plt.title(f'{section.capitalize()} Section - Extraction Counts by Model')
    plt.xlabel('Number of Extractions')
    plt.ylabel('Frequency')
    plt.xlim(0, max_values[section])
    plt.ylim(0, max_freq)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'extraction_hist_{section}_all_models.png', dpi=300)
    print(f"Saved histogram for {section} to: {plot_dir / f'extraction_hist_{section}_all_models.png'}")
    plt.close()

# 2. Histograms for BERT and Basic only
for section in sections:
    plt.figure(figsize=(12, 8))
    
    # Fixed bin width for consistent comparison
    bin_width = 5 if max_values[section] <= 50 else 10
    
    # Find max for just BERT and basic
    bert_basic_max = 0
    for model in ['basic', 'bert_anomaly']:
        col_name = f"{section}_{model}_extraction_count"
        if col_name in df.columns:
            val = df[col_name].quantile(0.995)
            bert_basic_max = max(bert_basic_max, val)
    
    # Round up for clean axes
    if bert_basic_max > 100:
        bert_basic_max = np.ceil(bert_basic_max / 10) * 10
    else:
        bert_basic_max = np.ceil(bert_basic_max / 5) * 5
        
    bins = np.arange(0, bert_basic_max + bin_width, bin_width)
    
    # Get max frequency for these models
    freqs = {}
    for model in ['basic', 'bert_anomaly']:
        col_name = f"{section}_{model}_extraction_count"
        if col_name in df.columns:
            header_col = f"has_{section}_headers"
            if header_col in df.columns:
                data = df[df[header_col] == 1][col_name]
            else:
                data = df[col_name]
            hist, _ = np.histogram(data, bins=bins)
            freqs[model] = max(hist)
    max_freq = max(freqs.values()) * 1.1 if freqs else 100
    
    for model in ['basic', 'bert_anomaly']:
        col_name = f"{section}_{model}_extraction_count"
        header_col = f"has_{section}_headers"
        
        if col_name in df.columns:
            if header_col in df.columns:
                section_df = df[df[header_col] == 1]
            else:
                section_df = df
            
            plt.hist(section_df[col_name], bins=bins, alpha=0.5, label=model)
    
    plt.title(f'{section.capitalize()} Section - BERT and Basic Models Only')
    plt.xlabel('Number of Extractions')
    plt.ylabel('Frequency')
    plt.xlim(0, bert_basic_max)
    plt.ylim(0, max_freq)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'extraction_hist_{section}_bert_basic.png', dpi=300)
    print(f"Saved BERT/basic histogram for {section}")
    plt.close()

# 2. Box plots for each section
for section in sections:
    plot_data = []
    plot_labels = []
    
    # Get adaptive range for this section
    plot_range = max_values[section]
    
    for model in models:
        col_name = f"{section}_{model}_extraction_count"
        header_col = f"has_{section}_headers"
        
        if col_name in df.columns:
            if header_col in df.columns:
                section_data = df[df[header_col] == 1][col_name]
            else:
                section_data = df[col_name]
            
            plot_data.append(section_data)
            plot_labels.extend([f"{section}_{model}"] * len(section_data))

    if plot_data:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=plot_data)
        plt.xticks(range(len(models)), [f"{model}" for model in models])
        plt.title(f'{section.capitalize()} Section - Extraction Counts Distribution')
        plt.ylabel('Number of Extractions')
        plt.xticks(rotation=45)
        
        # Set adaptive y-axis range
        plt.ylim(-plot_range * 0.05, plot_range)
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'extraction_boxplot_{section}.png', dpi=300, bbox_inches='tight')
        print(f"Saved boxplot for {section} to: {plot_dir / f'extraction_boxplot_{section}.png'}")
        plt.close()

# 3. Statistical Analysis with normalization - save to text file
with open(plot_dir / 'statistical_analysis.txt', 'w') as f:
    f.write("Statistical Analysis (Normalized by Section Presence):\n")
    f.write("-" * 50 + "\n")

    for section in sections:
        f.write(f"\n{section.upper()} SECTION:\n")
        header_col = f"has_{section}_headers"
        
        if header_col in df.columns:
            section_count = df[header_col].sum()
            f.write(f"Number of documents with {section} section: {section_count}\n")
        else:
            section_count = len(df)
            f.write(f"No header information for {section}, using total documents: {section_count}\n")
        
        for model in models:
            col_name = f"{section}_{model}_extraction_count"
            if col_name in df.columns:
                if header_col in df.columns:
                    section_df = df[df[header_col] == 1]
                else:
                    section_df = df
                
                stats = section_df[col_name].describe()
                proportion = section_df[col_name] / section_df['determination_count']
                
                f.write(f"\n{model.upper()}:\n")
                f.write(f"Mean extractions (in documents with {section}): {stats['mean']:.2f}\n")
                f.write(f"Median extractions: {stats['50%']:.2f}\n")
                f.write(f"Std dev: {stats['std']:.2f}\n")
                f.write(f"Mean proportion of determination_count: {proportion.mean():.2%}\n")

# 4. High Count Examples - save to separate text file
with open(plot_dir / 'high_count_examples.txt', 'w') as f:
    f.write("High Count Examples:\n")
    f.write("-" * 50 + "\n")

    for section in sections:
        header_col = f"has_{section}_headers"
        for model in models:
            count_col = f"{section}_{model}_extraction_count"
            text_col = f"{section}_{model}_extraction"
            
            if count_col in df.columns and text_col in df.columns:
                if header_col in df.columns:
                    section_df = df[df[header_col] == 1]
                else:
                    section_df = df
                
                mean_count = section_df[count_col].mean()
                std_count = section_df[count_col].std()
                threshold = mean_count + 2*std_count
                
                f.write(f"\nHigh count examples for {section}_{model}:\n")
                f.write(f"(Threshold: > {threshold:.2f} extractions)\n")
                
                high_counts = section_df[section_df[count_col] > threshold].sort_values(count_col, ascending=False)
                
                for idx, row in high_counts.head(3).iterrows():
                    f.write(f"\nDecision ID: {row['decisionID']}\n")
                    f.write(f"Count: {row[count_col]}\n")
                    f.write("First few extracted sentences:\n")
                    try:
                        sentences = ast.literal_eval(row[text_col])
                        for sent in sentences[:3]:
                            f.write(f"- {sent}\n")
                    except:
                        f.write("Could not parse extracted sentences\n")
                    f.write("-" * 30 + "\n")

print("\nAll outputs have been saved to the 'plots' directory:")
print(f"1. Histogram plot: {plot_dir / 'extraction_hist_cleaned_all_models.png'}")
print(f"2. Histogram plot: {plot_dir / 'extraction_hist_cleaned_bert_basic.png'}")
print(f"2. Boxplots: {plot_dir} /extraction_boxplot_*.png")
print(f"3. Statistical analysis: {plot_dir / 'statistical_analysis.txt'}")
print(f"4. High count examples: {plot_dir / 'high_count_examples.txt'}")
