import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluate_checkpoint import DeterminationDataset, evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

class EnsembleEvaluator:
    def __init__(self, checkpoint_configs, test_path, batch_size=32):
        """
        Args:
            checkpoint_configs: List of dicts, each containing:
                - 'path': path to checkpoint directory
                - 'threshold': classification threshold
                - 'name': model name for reference
            test_path: Path to test dataset
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_configs = checkpoint_configs
        self.test_path = test_path
        self.batch_size = batch_size
        
        # Load models and create dataloaders
        self.models = []
        self.dataloaders = []
        
        for config in checkpoint_configs:
            model_path = os.path.join(config['path'], 'best_model')
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.to(self.device)
            
            # Create dataset and dataloader
            dataset = DeterminationDataset(test_path, tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            
            self.models.append(model)
            self.dataloaders.append(dataloader)
            
        # Store raw data for later use
        self.raw_data = self.dataloaders[0].dataset.get_raw_data()
        self.texts = [item[0] for item in self.raw_data]
        self.true_labels = [item[1] for item in self.raw_data]

    def evaluate_all_models(self):
        """Evaluate all models and return their predictions"""
        all_predictions = []
        all_probabilities = []
        
        for i, (model, dataloader, config) in enumerate(zip(self.models, self.dataloaders, self.checkpoint_configs)):
            print(f"\nEvaluating model: {config['name']}")
            metrics, predictions, _, probs = evaluate(model, dataloader, self.device, config['threshold'])
            
            print(f"Model {config['name']} metrics:")
            print(f"F1: {metrics['f1']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            
            all_predictions.append(predictions)
            all_probabilities.append(probs)
            
        return all_predictions, all_probabilities

    def analyze_misclassifications(self, all_predictions):
        """Analyze misclassification patterns between models"""
        results = []
        
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                model1_name = self.checkpoint_configs[i]['name']
                model2_name = self.checkpoint_configs[j]['name']
                
                # Get incorrect predictions for both models
                incorrect1 = np.array(all_predictions[i]) != np.array(self.true_labels)
                incorrect2 = np.array(all_predictions[j]) != np.array(self.true_labels)
                
                # Calculate various intersection scenarios
                both_wrong = np.logical_and(incorrect1, incorrect2)
                only_model1_wrong = np.logical_and(incorrect1, ~incorrect2)
                only_model2_wrong = np.logical_and(~incorrect1, incorrect2)
                
                # Get counts
                both_wrong_count = int(np.sum(both_wrong))
                only_model1_count = int(np.sum(only_model1_wrong))
                only_model2_count = int(np.sum(only_model2_wrong))
                
                model1_total_errors = int(np.sum(incorrect1))
                model2_total_errors = int(np.sum(incorrect2))
                
                analysis = {
                    'model1': model1_name,
                    'model2': model2_name,
                    'model1_total_errors': model1_total_errors,
                    'model2_total_errors': model2_total_errors,
                    'both_wrong_count': both_wrong_count,
                    'only_model1_wrong': only_model1_count,
                    'only_model2_wrong': only_model2_count,
                }
                
                results.append(analysis)
        
        # Print detailed analysis
        print("\n=== Detailed Misclassification Analysis ===")
        for analysis in results:
            print(f"\nComparing {analysis['model1']} vs {analysis['model2']}:")
            print(f"Total samples with errors by either model: {analysis['only_model1_wrong'] + analysis['only_model2_wrong'] + analysis['both_wrong_count']}")
            print(f"- {analysis['model1']} total errors: {analysis['model1_total_errors']}")
            print(f"- {analysis['model2']} total errors: {analysis['model2_total_errors']}")
            print(f"Error breakdown:")
            print(f"- Both models wrong on same samples: {analysis['both_wrong_count']}")
            print(f"- Only {analysis['model1']} wrong: {analysis['only_model1_wrong']}")
            print(f"- Only {analysis['model2']} wrong: {analysis['only_model2_wrong']}")
        
        return pd.DataFrame(results)

    def majority_vote_prediction(self, all_predictions):
        """Compute majority vote predictions and metrics"""
        predictions_array = np.array(all_predictions)
        majority_predictions = np.mean(predictions_array, axis=0) >= 0.667  # 2/3 majority
        
        # Calculate metrics
        accuracy = accuracy_score(self.true_labels, majority_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.true_labels, majority_predictions, average='binary', zero_division=0
        )
        
        return {
            'predictions': majority_predictions,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }

    def save_results(self, all_predictions, all_probabilities, output_dir):
        """Save detailed results to Excel file"""
        results_dict = {
            'Text': self.texts,
            'True Label': self.true_labels
        }
        
        # Add individual model predictions and probabilities
        for i, config in enumerate(self.checkpoint_configs):
            results_dict[f"{config['name']}_prediction"] = all_predictions[i]
            results_dict[f"{config['name']}_probability"] = all_probabilities[i]
        
        # Add majority vote predictions
        majority_results = self.majority_vote_prediction(all_predictions)
        results_dict['Majority_Vote'] = majority_results['predictions']
        
        # Create DataFrame and save
        df = pd.DataFrame(results_dict)
        
        # Add error indicators
        for i, config in enumerate(self.checkpoint_configs):
            df[f"{config['name']}_error"] = df['True Label'] != df[f"{config['name']}_prediction"]
        
        df['Majority_Vote_Error'] = df['True Label'] != df['Majority_Vote']
        
        # Save to Excel
        output_path = os.path.join(output_dir, 'ensemble_results.xlsx')
        df.to_excel(output_path, index=False)
        print(f"\nSaved detailed results to: {output_path}")
        
        return df

def main():
    # Example usage
    checkpoint_configs = [
        {'path': '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/anomaly_run_20250330_171154', 'threshold': 0.29, 'name': 'anomaly'},
        {'path': '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/balanced_legal_roberta_base_20250331_162411', 'threshold': 0.23, 'name': 'balanced'},
        {'path': '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/high_recall_legal_roberta_base_20250331_155951', 'threshold': 0.2, 'name': 'recall'}
    ]
    
    test_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/test_dataset.json'
    output_dir = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/ensemble_results'
    
    evaluator = EnsembleEvaluator(checkpoint_configs, test_path)
    
    # Run evaluation
    all_predictions, all_probabilities = evaluator.evaluate_all_models()
    
    # Analyze misclassifications
    misclass_analysis = evaluator.analyze_misclassifications(all_predictions)
    print("\nMisclassification Analysis:")
    print(misclass_analysis)
    
    # Get majority vote results
    majority_results = evaluator.majority_vote_prediction(all_predictions)
    print("\nMajority Vote Metrics:")
    print(f"F1: {majority_results['metrics']['f1']:.4f}")
    print(f"Accuracy: {majority_results['metrics']['accuracy']:.4f}")
    print(f"Precision: {majority_results['metrics']['precision']:.4f}")
    print(f"Recall: {majority_results['metrics']['recall']:.4f}")
    
    # Save detailed results
    evaluator.save_results(all_predictions, all_probabilities, output_dir)

def analyze_ensemble_comparison(results_path, model_name='anomaly', output_dir='analysis_results'):
    """
    Analyze and compare ensemble results with a specific model's results.
    
    Args:
        results_path: Path to the ensemble_results.xlsx file
        model_name: Name of the model to compare with ensemble
        output_dir: Directory to save the PDF report
    """
    # Read the results
    df = pd.read_excel(results_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with PdfPages(os.path.join(output_dir, f'ensemble_vs_{model_name}_analysis.pdf')) as pdf:
        # 1. Create summary page
        plt.figure(figsize=(12, 8))
        plt.title(f'Comparison Summary: Ensemble vs {model_name}')
        
        # Calculate agreement statistics
        total_samples = len(df)
        both_correct = sum((df['Majority_Vote'] == df['True Label']) & 
                         (df[f'{model_name}_prediction'] == df['True Label']))
        both_wrong = sum((df['Majority_Vote'] != df['True Label']) & 
                       (df[f'{model_name}_prediction'] != df['True Label']))
        ensemble_only_correct = sum((df['Majority_Vote'] == df['True Label']) & 
                                 (df[f'{model_name}_prediction'] != df['True Label']))
        model_only_correct = sum((df['Majority_Vote'] != df['True Label']) & 
                               (df[f'{model_name}_prediction'] == df['True Label']))
        
        # Create summary text
        summary_text = (
            f"Total Samples: {total_samples}\n\n"
            f"Both Correct: {both_correct} ({both_correct/total_samples*100:.1f}%)\n"
            f"Both Wrong: {both_wrong} ({both_wrong/total_samples*100:.1f}%)\n"
            f"Only Ensemble Correct: {ensemble_only_correct} ({ensemble_only_correct/total_samples*100:.1f}%)\n"
            f"Only {model_name} Correct: {model_only_correct} ({model_only_correct/total_samples*100:.1f}%)\n"
        )
        
        plt.text(0.1, 0.5, summary_text, fontsize=12, va='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 2. Create confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        confusion_data = pd.crosstab(df['Majority_Vote'], df[f'{model_name}_prediction'],
                                   margins=True)
        sns.heatmap(confusion_data, annot=True, fmt='d', cmap='YlOrRd')
        plt.title(f'Prediction Comparison: Ensemble vs {model_name}')
        plt.xlabel(f'{model_name} Predictions')
        plt.ylabel('Ensemble Predictions')
        pdf.savefig()
        plt.close()
        
        # 3. Create disagreement analysis
        disagreements = df[df['Majority_Vote'] != df[f'{model_name}_prediction']]
        
        if len(disagreements) > 0:
            plt.figure(figsize=(12, 8))
            plt.title('Disagreement Analysis')
            
            # Calculate accuracy for disagreement cases
            ensemble_correct = sum(disagreements['Majority_Vote'] == disagreements['True Label'])
            model_correct = sum(disagreements[f'{model_name}_prediction'] == disagreements['True Label'])
            
            disagreement_text = (
                f"Total Disagreements: {len(disagreements)}\n\n"
                f"In disagreement cases:\n"
                f"Ensemble Correct: {ensemble_correct} ({ensemble_correct/len(disagreements)*100:.1f}%)\n"
                f"{model_name} Correct: {model_correct} ({model_correct/len(disagreements)*100:.1f}%)\n"
            )
            
            plt.text(0.1, 0.5, disagreement_text, fontsize=12, va='center')
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
        # 4. Save detailed disagreements table
        if len(disagreements) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Create table with relevant columns
            table_data = disagreements[['Text', 'True Label', 'Majority_Vote', 
                                      f'{model_name}_prediction']].head(20)
            
            table = ax.table(cellText=table_data.values,
                           colLabels=table_data.columns,
                           cellLoc='left',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.auto_set_column_width(col=list(range(len(table_data.columns))))
            
            plt.title('Sample Disagreements (First 20 cases)')
            pdf.savefig()
            plt.close()

if __name__ == "__main__":
    main() 