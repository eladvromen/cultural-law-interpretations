import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Type

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # trainers directory
project_root = os.path.dirname(os.path.dirname(current_dir))  # project root
determination_pipeline_dir = os.path.join(project_root, 'immigration_fine_tuning', 'determination', 'pipeline')

# Add determination pipeline directory to Python path
sys.path.insert(0, determination_pipeline_dir)

# Import ensemble evaluator from current directory
from evaluate_ensemble import EnsembleEvaluator

# Now we can import extractors since we added the pipeline directory to sys.path
from extractors.base.base_determination_extractor import BaseDeterminationExtractor
from extractors.basic.basic_determination_extraction import BasicDeterminationExtractor
from extractors.ngram.ngram_determination_extraction import NgramDeterminationExtractor

class HybridEnsembleEvaluator:
    def __init__(self, 
                 checkpoint_configs: List[Dict], 
                 test_path: str,
                 train_path: str,
                 batch_size: int = 32):
        """
        Args:
            checkpoint_configs: List of model checkpoint configurations
            test_path: Path to test dataset
            train_path: Path to training dataset (for extractors)
            batch_size: Batch size for model inference
        """
        # Initialize model ensemble evaluator
        self.model_ensemble = EnsembleEvaluator(checkpoint_configs, test_path, batch_size)
        self.test_path = test_path
        self.train_path = train_path
        
        # Initialize only Basic and Ngram extractors
        self.extractor_classes = [
            BasicDeterminationExtractor,
            NgramDeterminationExtractor
        ]
        
        self.extractors = []
        self._initialize_extractors()

    def _initialize_extractors(self):
        """Initialize and prepare all extractors"""
        print("\nInitializing extractors...")
        for extractor_class in self.extractor_classes:
            try:
                extractor = extractor_class()
                if hasattr(extractor, 'load_training_examples'):
                    try:
                        extractor.load_training_examples(self.train_path, self.test_path, use_chunking=True)
                    except TypeError:
                        extractor.load_training_examples(self.train_path, self.test_path)
                self.extractors.append(extractor)
                print(f"Successfully initialized {extractor_class.__name__}")
            except Exception as e:
                print(f"Failed to initialize {extractor_class.__name__}: {e}")

    def get_extractor_predictions(self):
        """Get predictions from all extractors"""
        print("\nGetting predictions from extractors...")
        extractor_predictions = []
        
        texts = self.model_ensemble.texts
        for extractor in self.extractors:
            predictions = []
            for text in texts:
                try:
                    determinations = extractor.extract_potential_determinations(text)
                    predictions.append(1 if determinations else 0)
                except Exception as e:
                    print(f"Error in {type(extractor).__name__} for text: {e}")
                    predictions.append(0)
            extractor_predictions.append(predictions)
            
        return extractor_predictions

    def evaluate_hybrid_ensemble(self):
        """Evaluate the hybrid ensemble combining both models and extractors"""
        # Get model predictions
        print("\nEvaluating neural models...")
        model_predictions, model_probabilities = self.model_ensemble.evaluate_all_models()
        
        # Get extractor predictions
        print("\nEvaluating extractors...")
        extractor_predictions = self.get_extractor_predictions()
        
        # Combine all predictions
        all_predictions = model_predictions + extractor_predictions
        predictions_array = np.array(all_predictions)
        
        # Calculate various voting thresholds
        voting_thresholds = [0.5, 0.6, 0.667, 0.75]
        voting_results = {}
        
        for threshold in voting_thresholds:
            majority_predictions = np.mean(predictions_array, axis=0) >= threshold
            metrics = self._calculate_metrics(majority_predictions)
            voting_results[f"threshold_{threshold}"] = metrics
        
        return {
            'model_predictions': model_predictions,
            'model_probabilities': model_probabilities,
            'extractor_predictions': extractor_predictions,
            'voting_results': voting_results
        }

    def _calculate_metrics(self, predictions):
        """Calculate metrics for given predictions"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        true_labels = self.model_ensemble.true_labels
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_results(self, results, output_dir):
        """Save detailed results to Excel file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        results_dict = {
            'Text': self.model_ensemble.texts,
            'True_Label': self.model_ensemble.true_labels
        }
        
        # Add model predictions
        for i, config in enumerate(self.model_ensemble.checkpoint_configs):
            results_dict[f"{config['name']}_prediction"] = results['model_predictions'][i]
            results_dict[f"{config['name']}_probability"] = results['model_probabilities'][i]
        
        # Add extractor predictions
        for i, extractor in enumerate(self.extractors):
            results_dict[f"{type(extractor).__name__}_prediction"] = results['extractor_predictions'][i]
        
        # Create DataFrame
        df = pd.DataFrame(results_dict)
        
        # Save results
        output_path = os.path.join(output_dir, 'hybrid_ensemble_results.xlsx')
        df.to_excel(output_path, index=False)
        print(f"\nSaved detailed results to: {output_path}")
        
        # Save voting results separately
        voting_df = pd.DataFrame(results['voting_results']).round(4)
        voting_path = os.path.join(output_dir, 'voting_results.csv')
        voting_df.to_csv(voting_path)
        print(f"Saved voting results to: {voting_path}")

def main():
    # Configuration
    checkpoint_configs = [
        {'path': '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/anomaly_run_20250330_171154', 'threshold': 0.29, 'name': 'anomaly'},
        {'path': '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/balanced_legal_roberta_base_20250331_162411', 'threshold': 0.23, 'name': 'balanced'},
        {'path': '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/high_recall_legal_roberta_base_20250331_155951', 'threshold': 0.2, 'name': 'recall'}
    ]
    
    test_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/test_dataset.json'
    train_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/train_dataset.json'
    output_dir = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models/hybrid_ensemble_results'
    
    # Initialize and run evaluation
    evaluator = HybridEnsembleEvaluator(checkpoint_configs, test_path, train_path)
    results = evaluator.evaluate_hybrid_ensemble()
    
    # Print voting results
    print("\nVoting Results:")
    for threshold, metrics in results['voting_results'].items():
        print(f"\n{threshold}:")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
    
    # Save results
    evaluator.save_results(results, output_dir)

if __name__ == "__main__":
    main() 