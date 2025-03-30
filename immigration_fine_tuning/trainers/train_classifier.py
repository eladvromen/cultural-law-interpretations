import os
import json
import torch
import numpy as np
from torch import nn
from datetime import datetime
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

def augment_text(text):
    # Simple augmentation by random word dropout
    words = text.split()
    if len(words) <= 3:  # Don't augment very short texts
        return text
    dropout_idx = random.randint(0, len(words)-1)
    words.pop(dropout_idx)
    return ' '.join(words)

class DeterminationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, training=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training = training
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        # Sort data by length for more efficient batching
        self.data = sorted(self.data, key=lambda x: len(x['text'].split()))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # 30% chance of augmentation during training
        if self.training and random.random() < 0.3:
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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
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

def evaluate(model, dataloader, device, threshold=0.3):
    model.eval()
    predictions = []
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
            # Use custom threshold on the positive class probability
            preds = (probs[:, 1] > threshold).long()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }

def main():
    # Hyperparameters
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    ACCUMULATION_STEPS = 6
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    TRAIN_VAL_SPLIT = 0.15
    WEIGHT_DECAY = 0.1
    DROPOUT_RATE = 0.2
    
    # Create timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(MODEL_SAVE_DIR, f'run_{timestamp}')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(model_save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump({
            'max_length': MAX_LENGTH,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'warmup_ratio': WARMUP_RATIO,
            'train_val_split': TRAIN_VAL_SPLIT,
            'weight_decay': WEIGHT_DECAY,
            'dropout_rate': DROPOUT_RATE
        }, f, indent=2)

    # Paths
    train_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/train_dataset.json'
    test_path = '/data/shil6369/cultural-law-interpretations/immigration_fine_tuning/determination/pipeline/data/test_dataset.json'
    
    # Device and memory info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB used")
    
    # Load tokenizer and model
    print("Loading LegalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        num_labels=2,
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE,
        classifier_dropout=DROPOUT_RATE,
        problem_type="single_label_classification",
       
    )
    model.to(device)
    
    # Create datasets
    print("Loading datasets...")
    full_dataset = DeterminationDataset(train_path, tokenizer, MAX_LENGTH, training=True)
    test_dataset = DeterminationDataset(test_path, tokenizer, MAX_LENGTH, training=False)
    
    # Split training data into train and validation
    train_size = int((1 - TRAIN_VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders with consistent configurations
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True
    )
    
    # Use a more robust optimizer configuration
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE * 10}
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize focal loss
    criterion = FocalLoss(
        alpha=1.0,  # Balance parameter
        gamma=2.0   # Focusing parameter - higher values increase focus on hard examples
    ).to(device)
    
    # Improved early stopping
    patience = 3  # Increased patience
    best_val_f1 = 0
    patience_counter = 0
    min_delta = 0.001  # Minimum improvement required
    
    # Training loop
    print("Starting training...")
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Train with focal loss
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
        
        # Validate
        val_metrics = evaluate(model, val_dataloader, device)
        print(f"Validation metrics:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"F1 Score: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            model_save_path = os.path.join(model_save_dir, 'best_model')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"Saved best model with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # After training completes, find best threshold
    print("\nTuning classification threshold...")
    thresholds = [0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50]
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        val_metrics = evaluate(model, val_dataloader, device, threshold=threshold)
        print(f"\nThreshold: {threshold}")
        print(f"F1: {val_metrics['f1']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall: {val_metrics['recall']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_threshold = threshold
    
    # Use best threshold for final evaluation
    print(f"\nEvaluating on test set with threshold = {best_threshold}...")
    test_metrics = evaluate(model, test_dataloader, device, threshold=best_threshold)
    
    # Save test results
    with open(os.path.join(model_save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

if __name__ == "__main__":
    main() 