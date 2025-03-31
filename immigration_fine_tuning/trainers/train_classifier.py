import os
import json
import torch
import numpy as np
import pandas as pd
from torch import nn
from datetime import datetime
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

# Set HF cache directory
os.environ["HF_HOME"] = "/data/resource/huggingface"

# Set all random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Replace existing random seed setting with this function call
set_seed(42)

# Create model save directory if it doesn't exist
MODEL_SAVE_DIR = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Update augmentation constants
TEXT_AUGMENTATION_PROB = 0.15  # Random word dropout probability
RANDOM_SWAP_PROB = 0.10  # Random word swapping probability

def augment_text(text):
    """
    Apply text augmentation techniques:
    1. Random word dropout
    2. Random word swapping
    """
    words = text.split()
    
    # Don't augment very short texts
    if len(words) <= 3:
        return text
    
    # Random word dropout
    if random.random() < TEXT_AUGMENTATION_PROB:
        dropout_idx = random.randint(0, len(words)-1)
        words.pop(dropout_idx)
    
    # Random word swapping
    if random.random() < RANDOM_SWAP_PROB and len(words) >= 2:
        idx1 = random.randint(0, len(words)-1)
        idx2 = random.randint(0, len(words)-1)
        # Make sure we're swapping different words
        while idx1 == idx2:
            idx2 = random.randint(0, len(words)-1)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return ' '.join(words)

class DeterminationDataset(Dataset):
    def __init__(self, data_source, tokenizer, max_length=128, training=False, augmentation_rate=0.3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training
        self.augmentation_rate = augmentation_rate
        
        # Handle either direct data list or file path
        if isinstance(data_source, str):
            # It's a file path
            with open(data_source, 'r') as f:
                self.data = json.load(f)
        else:
            # It's already a data list
            self.data = data_source
            
        # Sort data by length for more efficient batching
        self.data = sorted(self.data, key=lambda x: len(x['text'].split()))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # Apply augmentation during training with configurable probability
        if self.training and random.random() < self.augmentation_rate:
            text = augment_text(text)
        
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha (class weights) correctly based on the target class of each sample
        if isinstance(self.alpha, torch.Tensor) and len(self.alpha) > 1:
            # If alpha is a tensor with class weights, gather the weight for each sample's target class
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            # If alpha is a scalar, apply it uniformly
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps, criterion):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use focal loss instead of default cross entropy
        loss = criterion(outputs.logits, labels) / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, threshold=0.5, return_predictions=False):
    model.eval()
    predictions = []
    true_labels = []
    all_probs = []
    texts = [] if return_predictions else None  # Only collect texts if needed
    
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
            preds = (probs[:, 1] > threshold).long()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # If we need to return predictions, also gather the original texts
            # This requires modifying the dataset and dataloader to include texts
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }
    
    if return_predictions:
        return metrics, predictions, true_labels, all_probs
    return metrics

# Define model-specific hyperparameter sets
def get_model_hyperparameters(model_name):
    """
    Return hyperparameters optimized for specific model architectures.
    """
    # Default hyperparameters (for LegalBERT)
    default_params = {
        'max_length': 256,
        'batch_size': 32,
        'accumulation_steps': 4,
        'epochs': 7,
        'learning_rate': 8e-6,
        'warmup_ratio': 0.05,
        'train_val_split': 0.15,
        'weight_decay': 0.015,
        'dropout_rate': 0.25,
        'fixed_gamma': 2.0,
        'augmentation_rate': 0.3,
        'class_weight_enabled': True,
        'recall_focus': True
    }
    
    # Model-specific hyperparameter adjustments
    if 'roberta' in model_name.lower():
        return {
            'max_length': 256,
            'batch_size': 24,  # Smaller batch size as RoBERTa is bigger
            'accumulation_steps': 4,
            'epochs': 6,
            'learning_rate': 5e-6,  # Lower learning rate for RoBERTa
            'warmup_ratio': 0.1,    # More warmup for RoBERTa
            'train_val_split': 0.15,
            'weight_decay': 0.01,
            'dropout_rate': 0.2,
            'fixed_gamma': 2.0,
            'augmentation_rate': 0.3,
            'class_weight_enabled': True,
            'recall_focus': True
        }
    elif 'deberta' in model_name.lower():
        return {
            'max_length': 256,
            'batch_size': 16,  # Smaller batch size as DeBERTa is bigger
            'accumulation_steps': 8,  # More accumulation to compensate for smaller batch
            'epochs': 5,
            'learning_rate': 3e-6,  # Even lower learning rate for DeBERTa
            'warmup_ratio': 0.1,
            'train_val_split': 0.15,
            'weight_decay': 0.02,
            'dropout_rate': 0.15,
            'fixed_gamma': 1.5,
            'augmentation_rate': 0.3,
            'class_weight_enabled': True,
            'recall_focus': True
        }
    else:
        # Return default parameters for other models including LegalBERT
        return default_params

def main(model_name="nlpaueb/legal-bert-base-uncased"):
    # Get model-specific hyperparameters
    hp = get_model_hyperparameters(model_name)
    
    # Extract hyperparameters
    MAX_LENGTH = hp['max_length']
    BATCH_SIZE = hp['batch_size']
    ACCUMULATION_STEPS = hp['accumulation_steps']
    EPOCHS = hp['epochs']
    LEARNING_RATE = hp['learning_rate']
    WARMUP_RATIO = hp['warmup_ratio']
    TRAIN_VAL_SPLIT = hp['train_val_split']
    WEIGHT_DECAY = hp['weight_decay']
    DROPOUT_RATE = hp['dropout_rate']
    FIXED_GAMMA = hp['fixed_gamma']
    AUGMENTATION_RATE = hp['augmentation_rate']
    CLASS_WEIGHT_ENABLED = hp['class_weight_enabled']
    RECALL_FOCUS = hp['recall_focus']
    
    # Paths - keep only original train data
    train_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/train_dataset.json'
    test_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/test_dataset.json'
    
    # Device and memory info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB used")
        torch.cuda.empty_cache()
    
    # Extract model short name for directory naming
    model_short_name = model_name.split('/')[-1].replace('-', '_')
    
    # Print selected hyperparameters
    print(f"\nTraining {model_name} with the following hyperparameters:")
    for param, value in hp.items():
        print(f"  {param}: {value}")
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load test dataset
    test_dataset = DeterminationDataset(test_path, tokenizer, MAX_LENGTH, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    # Create timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(MODEL_SAVE_DIR, f"{model_short_name}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(model_save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump({
            'model_name': model_name,
            'max_length': MAX_LENGTH,
            'batch_size': BATCH_SIZE,
            'accumulation_steps': ACCUMULATION_STEPS,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'warmup_ratio': WARMUP_RATIO,
            'train_val_split': TRAIN_VAL_SPLIT,
            'weight_decay': WEIGHT_DECAY,
            'dropout_rate': DROPOUT_RATE,
            'gamma': FIXED_GAMMA,
            'class_weight_enabled': CLASS_WEIGHT_ENABLED,
            'recall_focus': RECALL_FOCUS,
            'augmentation_rate': AUGMENTATION_RATE,
            'word_dropout_prob': TEXT_AUGMENTATION_PROB,
            'word_swap_prob': RANDOM_SWAP_PROB,
            'train_data_path': train_path
        }, f, indent=2)
    
    # Load training data
    print("Loading training data...")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    print(f"Total training samples: {len(train_data)}")
    
    # Analyze class distribution
    labels = [sample['label'] for sample in train_data]
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"Class distribution in training data:")
    for label, count in class_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Calculate class weights if enabled
    if CLASS_WEIGHT_ENABLED:
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
        print(f"Computed class weights: {class_weights}")
        
        # Apply smoothing to avoid extreme values
        class_weights = np.clip(class_weights, 0.5, 2.0)
        print(f"Smoothed class weights: {class_weights}")
        
        # Convert to tensor for loss function
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    else:
        class_weights_tensor = None
    
    # Create dataset with augmentation rate
    full_dataset = DeterminationDataset(
        train_data, 
        tokenizer, 
        MAX_LENGTH, 
        training=True,
        augmentation_rate=AUGMENTATION_RATE
    )
    
    # Split training data into train and validation
    train_size = int((1 - TRAIN_VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    # Initialize model
    print(f"Initializing model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE,
        classifier_dropout=DROPOUT_RATE,
        problem_type="single_label_classification",
    )
    model.to(device)
    
    # Adjust optimizer for different model architectures
    # Check if the model has a 'bert' attribute, otherwise look for 'roberta' or use the base model
    if hasattr(model, 'bert'):
        base_params = model.bert.parameters()
        base_param_name = "bert"
    elif hasattr(model, 'roberta'):
        base_params = model.roberta.parameters()
        base_param_name = "roberta"
    else:
        # Fallback for other architectures
        # Exclude classifier parameters for separate learning rate
        classifier_params_ids = [id(p) for p in model.classifier.parameters()]
        base_params = [p for p in model.parameters() if id(p) not in classifier_params_ids]
        base_param_name = "base model"
        
    # Initialize optimizer with reduced classifier learning rate
    print(f"Setting up optimizer with different learning rates for {base_param_name} and classifier")
    optimizer = AdamW(
        [
            {"params": base_params, "lr": LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE * 3}
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )
    
    # Calculate steps with accumulation
    total_steps = len(train_dataloader) * EPOCHS // ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize loss function
    if CLASS_WEIGHT_ENABLED and class_weights_tensor is not None:
        print("Using weighted focal loss")
        criterion = FocalLoss(
            alpha=class_weights_tensor,
            gamma=FIXED_GAMMA
        ).to(device)
    else:
        criterion = FocalLoss(
            alpha=1.0,
            gamma=FIXED_GAMMA
        ).to(device)
    
    # Training loop
    patience = 4
    patience_counter = 0
    best_val_metric = 0
    best_model_state = None
    
    print(f"\nStarting training with Gamma = {FIXED_GAMMA}")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            ACCUMULATION_STEPS,
            criterion
        )
        print(f"Training loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_dataloader, device)
        print(f"Validation metrics:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall: {val_metrics['recall']:.4f}")
        print(f"F1 Score: {val_metrics['f1']:.4f}")
        
        # Decide which metric to track for early stopping
        tracked_metric = val_metrics['recall'] if RECALL_FOCUS else val_metrics['f1']
        metric_name = "recall" if RECALL_FOCUS else "F1"
        
        # Check for improvement and apply early stopping
        if tracked_metric > best_val_metric:
            best_val_metric = tracked_metric
            patience_counter = 0
            print(f"New best validation {metric_name}: {best_val_metric:.4f}. Saving model state...")
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"Validation {metric_name} did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    
    # Load the best model state found during training
    if best_model_state:
        print(f"\nLoading best model state with {metric_name}: {best_val_metric:.4f}")
        model.load_state_dict(best_model_state)
    
    # Find best threshold based on optimization goal
    print("\nTuning classification threshold...")
    if RECALL_FOCUS:
        # Use lower thresholds if focusing on recall
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        best_recall = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            val_metrics = evaluate(model, val_dataloader, device, threshold=threshold)
            f1 = val_metrics['f1']
            recall = val_metrics['recall']
            precision = val_metrics['precision']
            
            print(f"Threshold: {threshold:.2f} -> Recall: {recall:.4f}, F1: {f1:.4f}, P: {precision:.4f}")
            
            # Only accept thresholds where precision isn't terrible
            if recall > best_recall and precision >= 0.5:
                best_recall = recall
                best_threshold = threshold
        
        print(f"\nBest threshold for recall: {best_threshold:.2f} with recall: {best_recall:.4f}")
    else:
        # Use standard thresholds if optimizing for F1
        thresholds = [0.37, 0.40, 0.44, 0.47, 0.50, 0.53, 0.56, 0.60]
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            val_metrics = evaluate(model, val_dataloader, device, threshold=threshold)
            print(f"Threshold: {threshold:.2f} -> F1: {val_metrics['f1']:.4f}, P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_threshold = threshold
        
        print(f"\nBest threshold for F1: {best_threshold:.2f} with F1: {best_f1:.4f}")
    
    # Save best model
    model_save_path = os.path.join(model_save_dir, 'best_model')
    print(f"Saving best model to {model_save_path}")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Final evaluation with test set
    print(f"\nEvaluating on test set with threshold = {best_threshold:.2f}...")
    test_metrics, predictions, true_labels, probs = evaluate(model, test_dataloader, device, threshold=best_threshold, return_predictions=True)
    
    print(f"Test metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # Analyze error patterns
    false_negatives = 0
    false_positives = 0
    
    for pred, true in zip(predictions, true_labels):
        if pred == 0 and true == 1:
            false_negatives += 1
        elif pred == 1 and true == 0:
            false_positives += 1
    
    print(f"False Negatives: {false_negatives}/{len(predictions)} ({false_negatives/len(predictions)*100:.2f}%)")
    print(f"False Positives: {false_positives}/{len(predictions)} ({false_positives/len(predictions)*100:.2f}%)")
    
    # Save test results
    with open(os.path.join(model_save_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'threshold': best_threshold,
            'gamma': FIXED_GAMMA,
            'false_negatives': false_negatives,
            'false_positives': false_positives,
            'class_weight_enabled': CLASS_WEIGHT_ENABLED,
            'recall_focused': RECALL_FOCUS
        }, f, indent=2)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for legal text classification.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="nlpaueb/legal-bert-base-uncased",
        help="HuggingFace model identifier (e.g., 'nlpaueb/legal-bert-base-uncased', 'saibo/legal-roberta-base')"
    )
    args = parser.parse_args()
    
    # Pass model name to main function
    main(model_name=args.model_name) 