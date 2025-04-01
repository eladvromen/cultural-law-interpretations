import os
import json
import torch
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Dataset Class (copied from train_classifier.py) ---
class DeterminationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path # Store path for later loading raw data
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
    def get_raw_data(self):
        # Helper to get original text and labels
        return [(item['text'], item['label']) for item in self.data]

# --- Evaluation Function (adapted from train_classifier.py) ---
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    predictions = []
    true_labels = []
    all_probs = [] # Store probabilities
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs.logits, dim=1)
            # Use custom threshold on the positive class probability
            preds = (probs[:, 1] > threshold).long()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Store only positive class probability
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }
    
    # Return metrics, predictions, true labels, and probabilities
    return metrics, predictions, true_labels, all_probs

# --- Add Threshold Optimization Function ---
def optimize_threshold(model, dataloader, device, metric_to_optimize='f1'):
    """
    Try different thresholds and find the optimal one based on specified metric.
    
    Args:
        model: The trained model
        dataloader: Test dataloader
        device: Computation device (CPU/GPU)
        metric_to_optimize: Which metric to optimize ('f1', 'accuracy', 'precision', or 'recall')
        
    Returns:
        best_threshold: The optimal threshold
        best_metrics: Metrics at the optimal threshold
        all_thresholds_results: Results for all tried thresholds
    """
    # Get all probabilities once
    model.eval()
    all_probs = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs.logits, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())  # Store only positive class probabilities
            true_labels.extend(labels.cpu().numpy())
    
    # Try different thresholds
    thresholds = [i/100 for i in range(20, 50)]  # 0.01 to 0.99 in steps of 0.01
    best_score = -1
    best_threshold = 0.5
    best_metrics = None
    all_thresholds_results = []
    
    for threshold in thresholds:
        # Apply threshold
        predictions = [1 if prob > threshold else 0 for prob in all_probs]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        all_thresholds_results.append(metrics)
        
        # Update best threshold based on the metric to optimize
        current_score = metrics[metric_to_optimize]
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics, all_thresholds_results

def run_targeted_evaluation(model, test_dataloader, device, threshold, output_path):
    """Run evaluation with a specific threshold and save false predictions"""
    print(f"\nRunning targeted evaluation with threshold: {threshold}")
    metrics, predictions, true_labels, all_probs = evaluate(model, test_dataloader, device, threshold=threshold)

    print("\n--- Test Set Metrics ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Get raw data and create results DataFrame
    raw_data_tuples = test_dataloader.dataset.get_raw_data()
    raw_texts = [item[0] for item in raw_data_tuples]

    if not (len(raw_texts) == len(true_labels) == len(predictions) == len(all_probs)):
        raise ValueError("Mismatch in lengths of data lists!")

    results_df = pd.DataFrame({
        'Text': raw_texts,
        'True Label': true_labels,
        'Predicted Label': predictions,
        'Predicted Probability (Class 1)': all_probs
    })

    false_predictions_df = results_df[results_df['True Label'] != results_df['Predicted Label']].copy()
    false_predictions_df['Error Type'] = false_predictions_df.apply(
        lambda row: 'False Positive' if row['Predicted Label'] == 1 else 'False Negative', axis=1
    )

    print(f"Found {len(false_predictions_df)} incorrect predictions.")

    try:
        false_predictions_df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Saved false predictions to: {output_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        csv_path = output_path.replace('.xlsx', '.csv')
        try:
            false_predictions_df.to_csv(csv_path, index=False)
            print(f"Saved false predictions to: {csv_path}")
        except Exception as csv_e:
            print(f"Error saving CSV file: {csv_e}")

    return metrics

def main(args):
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Hyperparameters ---
    hyperparams_path = os.path.join(args.checkpoint_dir, 'hyperparameters.json')
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        max_length = hyperparams.get('max_length', 256)
    else:
        print("Warning: hyperparameters.json not found. Using default max_length=256")
        max_length = 256

    # --- Load Model and Tokenizer ---
    model_path = os.path.join(args.checkpoint_dir, 'best_model')
    print(f"Loading model and tokenizer from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # --- Load Test Data ---
    print(f"Loading test data from: {args.test_path}")
    test_dataset = DeterminationDataset(args.test_path, tokenizer, max_length=max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    if args.optimize_threshold:
        # Run threshold optimization
        print(f"\nOptimizing threshold based on {args.metric_to_optimize}...")
        best_threshold, best_metrics, all_thresholds_results = optimize_threshold(
            model, test_dataloader, device, metric_to_optimize=args.metric_to_optimize
        )
        
        # Save threshold optimization results
        results_path = os.path.join(args.checkpoint_dir, 'threshold_optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                "best_threshold": best_threshold,
                "best_metrics": best_metrics,
                "all_results": all_thresholds_results
            }, f, indent=2)
        print(f"Saved threshold optimization results to: {results_path}")
        
        # Run evaluation with optimal threshold
        threshold = best_threshold
    else:
        # Use provided threshold for targeted evaluation
        threshold = args.threshold

    # Run evaluation and save false predictions
    output_path = os.path.join(args.checkpoint_dir, f'false_predictions_thresh_{threshold:.3f}.xlsx')
    run_targeted_evaluation(model, test_dataloader, device, threshold, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint with threshold optimization or targeted evaluation.")
    parser.add_argument("checkpoint_dir", type=str, help="Path to the checkpoint directory")
    parser.add_argument("--test_path", type=str, 
                        default='/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/test_dataset.json',
                        help="Path to test dataset")
    parser.add_argument("--optimize_threshold", action="store_true", 
                        help="Whether to run threshold optimization")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold to use for targeted evaluation (ignored if optimize_threshold is True)")
    parser.add_argument("--metric_to_optimize", type=str, default='f1',
                        choices=['f1', 'accuracy', 'precision', 'recall'],
                        help="Metric to optimize when finding best threshold")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    
    args = parser.parse_args()
    main(args)

    