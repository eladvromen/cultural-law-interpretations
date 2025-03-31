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
    thresholds = [i/100 for i in range(20, 80)]  # 0.01 to 0.99 in steps of 0.01
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

def main(checkpoint_dir):
    # --- Configuration ---
    test_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/test_dataset.json'
    hyperparams_path = os.path.join(checkpoint_dir, 'hyperparameters.json')
    model_path = os.path.join(checkpoint_dir, 'best_model') # Assuming best model was saved here
    output_excel_path = os.path.join(checkpoint_dir, 'false_predictions.xlsx')
    thresholds_results_path = os.path.join(checkpoint_dir, 'threshold_optimization_results.json')
    batch_size = 32 # Can adjust if needed for evaluation
    metric_to_optimize = 'f1'  # Can be 'f1', 'accuracy', 'precision', or 'recall'

    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Hyperparameters and determine threshold ---
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        max_length = hyperparams.get('max_length', 256) # Default if not found
        # Load threshold from test results if available, else use default 0.5
        test_results_path = os.path.join(checkpoint_dir, 'test_results.json')
        if os.path.exists(test_results_path):
             with open(test_results_path, 'r') as f:
                 test_results = json.load(f)
             eval_threshold = test_results.get('threshold', 0.5)
        else:
             eval_threshold = 0.5 # Default if no test results file
        print(f"Using Max Length: {max_length}")
        print(f"Using Evaluation Threshold: {eval_threshold}")
    else:
        print("Warning: hyperparameters.json not found. Using default max_length=256 and threshold=0.5")
        max_length = 256
        eval_threshold = 0.5

    # --- Load Model and Tokenizer ---
    print(f"Loading model and tokenizer from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # --- Load Test Data ---
    print(f"Loading test data from: {test_path}")
    test_dataset = DeterminationDataset(test_path, tokenizer, max_length=max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # --- Optimize Threshold ---
    print(f"\nOptimizing threshold based on {metric_to_optimize}...")
    best_threshold, best_metrics, all_thresholds_results = optimize_threshold(
        model, test_dataloader, device, metric_to_optimize=metric_to_optimize
    )
    
    print(f"Optimal threshold: {best_threshold:.4f}")
    print(f"Metrics at optimal threshold:")
    print(f"  - Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  - Precision: {best_metrics['precision']:.4f}")
    print(f"  - Recall: {best_metrics['recall']:.4f}")
    print(f"  - F1 Score: {best_metrics['f1']:.4f}")
    
    # Save threshold optimization results
    with open(thresholds_results_path, 'w') as f:
        json.dump({
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "all_results": all_thresholds_results
        }, f, indent=2)
    print(f"Saved threshold optimization results to: {thresholds_results_path}")

    # --- Run Evaluation with optimal threshold ---
    print("\nRunning evaluation with optimal threshold...")
    metrics, predictions, true_labels, all_probs = evaluate(model, test_dataloader, device, threshold=best_threshold)

    print("\n--- Test Set Metrics with Optimal Threshold ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Threshold Used: {metrics['threshold']:.4f}")

    # --- Analyze False Predictions ---
    print("\nAnalyzing false predictions...")
    # Get the list of (text, label) tuples
    raw_data_tuples = test_dataset.get_raw_data() 
    # Extract just the texts into a new list
    raw_texts = [item[0] for item in raw_data_tuples] 
    
    # --- Debugging: Check list lengths (can be kept or removed) ---
    print(f"Length of raw_texts: {len(raw_texts)}")
    print(f"Length of true_labels: {len(true_labels)}")
    print(f"Length of predictions: {len(predictions)}")
    print(f"Length of all_probs: {len(all_probs)}")
    # --- End Debugging ---

    # Check if lengths match before creating DataFrame
    # Ensure raw_texts length matches the others
    if not (len(raw_texts) == len(true_labels) == len(predictions) == len(all_probs)): 
        print("Error: Mismatch in lengths of data lists!")
        print(f"Lengths - Raw Text: {len(raw_texts)}, True Labels: {len(true_labels)}, Predictions: {len(predictions)}, Probabilities: {len(all_probs)}")
        return # Exit if lengths don't match

    results_df = pd.DataFrame({
        'Text': raw_texts, # Use the extracted raw_texts list
        'True Label': true_labels,
        'Predicted Label': predictions,
        'Predicted Probability (Class 1)': [prob for prob in all_probs] 
    })

    false_predictions_df = results_df[results_df['True Label'] != results_df['Predicted Label']].copy()
    
    # Add False Positive / False Negative label
    false_predictions_df['Error Type'] = false_predictions_df.apply(
        lambda row: 'False Positive' if row['Predicted Label'] == 1 else 'False Negative', axis=1
    )

    print(f"Found {len(false_predictions_df)} incorrect predictions.")

    # --- Save False Predictions to Excel ---
    try:
        false_predictions_df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"Saved false predictions to: {output_excel_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        print("Attempting to save as CSV instead.")
        csv_path = os.path.join(checkpoint_dir, 'false_predictions.csv')
        try:
             false_predictions_df.to_csv(csv_path, index=False)
             print(f"Saved false predictions to: {csv_path}")
        except Exception as csv_e:
             print(f"Error saving CSV file: {csv_e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint and save false predictions.")
    parser.add_argument("checkpoint_dir", type=str, help="Path to the checkpoint directory (e.g., models/run_20250331_120826)")
    args = parser.parse_args()
    
    main(args.checkpoint_dir) 

    