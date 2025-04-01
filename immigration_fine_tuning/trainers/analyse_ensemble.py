import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_ensemble_comparison(results_path, model_name='anomaly', output_dir='analysis_results'):
    """
    Analyze and compare ensemble results with a specific model's results.
    
    Args:
        results_path: Path to the ensemble_results.xlsx file
        model_name: Name of the model to compare with ensemble
        output_dir: Directory to save analysis files
    """
    # Read the results
    df = pd.read_excel(results_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate agreement statistics
    df['agreement'] = df['Majority_Vote'] == df[f'{model_name}_prediction']
    df['ensemble_correct'] = df['Majority_Vote'] == df['True Label']
    df[f'{model_name}_correct'] = df[f'{model_name}_prediction'] == df['True Label']
    
    # Add analysis columns to DataFrame
    df['prediction_status'] = 'both_correct'
    df.loc[(~df['ensemble_correct']) & (~df[f'{model_name}_correct']), 'prediction_status'] = 'both_wrong'
    df.loc[(df['ensemble_correct']) & (~df[f'{model_name}_correct']), 'prediction_status'] = 'only_ensemble_correct'
    df.loc[(~df['ensemble_correct']) & (df[f'{model_name}_correct']), 'prediction_status'] = 'only_model_correct'
    
    # Save detailed CSV
    csv_columns = [
        'Text', 
        'True Label', 
        'Majority_Vote', 
        f'{model_name}_prediction',
        f'{model_name}_probability',
        'agreement',
        'ensemble_correct',
        f'{model_name}_correct',
        'prediction_status'
    ]
    
    df[csv_columns].to_csv(os.path.join(output_dir, f'ensemble_vs_{model_name}_analysis.csv'), index=False)
    
    # 1. Create confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    confusion_data = pd.crosstab(df['Majority_Vote'], df[f'{model_name}_prediction'],
                               margins=True)
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='YlOrRd')
    plt.title(f'Prediction Comparison: Ensemble vs {model_name}')
    plt.xlabel(f'{model_name} Predictions')
    plt.ylabel('Ensemble Predictions')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Create prediction status distribution plot
    plt.figure(figsize=(10, 6))
    status_counts = df['prediction_status'].value_counts()
    status_counts.plot(kind='bar')
    plt.title('Distribution of Prediction Outcomes')
    plt.xlabel('Prediction Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_status_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Create probability distribution plot for agreements vs disagreements
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['agreement']], x=f'{model_name}_probability', label='Agreements')
    sns.kdeplot(data=df[~df['agreement']], x=f'{model_name}_probability', label='Disagreements')
    plt.title(f'Probability Distribution: {model_name} Model')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\nAnalysis Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Agreement cases: {sum(df['agreement'])} ({sum(df['agreement'])/len(df)*100:.1f}%)")
    print(f"Disagreement cases: {sum(~df['agreement'])} ({sum(~df['agreement'])/len(df)*100:.1f}%)")
    print(f"\nFiles saved in: {output_dir}")
    print(f"- ensemble_vs_{model_name}_analysis.csv")
    print("- confusion_matrix.png")
    print("- prediction_status_distribution.png")
    print("- probability_distribution.png")

def main():
    results_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/ensemble_results/ensemble_results.xlsx'
    output_dir = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/ensemble_results/analysis'
    
    analyze_ensemble_comparison(results_path, model_name='anomaly', output_dir=output_dir)

if __name__ == "__main__":
    main()