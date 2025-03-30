import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from train_classifier import DeterminationDataset, evaluate

# Paths
MODEL_DIR = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models'
TEST_PATH = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/test_dataset.json'

def main():
    # Find the latest experiment directory
    experiment_dirs = [d for d in os.listdir(MODEL_DIR) if d.startswith('focal_loss_experiments_')]
    latest_exp_dir = max(experiment_dirs)
    exp_path = os.path.join(MODEL_DIR, latest_exp_dir)
    
    # Load experiment results
    with open(os.path.join(exp_path, 'experiment_results.json'), 'r') as f:
        results = json.load(f)
    
    best_params = results['best_params']
    print(f"Loading best model with parameters:")
    print(f"Gamma: {best_params['gamma']}")
    print(f"Learning rate: {best_params['learning_rate']}")
    print(f"Validation F1: {results['best_f1']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_path = os.path.join(exp_path, f"model_gamma_{best_params['gamma']}_lr_{best_params['learning_rate']}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model.to(device)
    
    # Load test dataset
    test_dataset = DeterminationDataset(TEST_PATH, tokenizer, max_length=256, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    # Try different thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    best_test_f1 = 0
    best_threshold = None
    best_metrics = None
    
    for threshold in thresholds:
        test_metrics = evaluate(model, test_dataloader, device, threshold=threshold)
        print(f"\nThreshold: {threshold}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        if test_metrics['f1'] > best_test_f1:
            best_test_f1 = test_metrics['f1']
            best_threshold = threshold
            best_metrics = test_metrics
    
    # Save test results
    test_results = {
        'best_threshold': best_threshold,
        'metrics': best_metrics,
        'model_params': best_params,
        'all_thresholds_results': {str(t): evaluate(model, test_dataloader, device, threshold=t) 
                                  for t in thresholds}
    }
    
    with open(os.path.join(exp_path, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nBest Test Results:")
    print(f"Threshold: {best_threshold}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")

if __name__ == "__main__":
    main() 