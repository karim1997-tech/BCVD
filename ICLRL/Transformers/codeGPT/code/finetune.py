import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2ForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef, 
    confusion_matrix
)
from torch.optim import AdamW
from tqdm import tqdm
import csv
from collections import defaultdict
from datetime import datetime
import time
import GPUtil

class Config:
    MODEL_NAMES = {
        'codegpt': 'microsoft/CodeGPT-small-py'  # Using CodeGPT-small-py variant
    }
    
    # Training parameters
    BASE_SEED = 1500  # Base seed to generate other seeds
    BATCH_SIZE = 16  # Reduced batch size for GPT models
    MAX_LEN = 512
    EPOCHS = 30
    LEARNING_RATE = 2e-5
    EARLY_STOPPING_PATIENCE = 5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    
    # Paths
    DATA_PATH = "E:/models/workB/ICLRL/data/assembly2025.csv"
    BASE_SAVE_DIR = "E:/models/workB/ICLRL/Transformers/codegpt/results"
    
    @staticmethod
    def get_config(seed):
        config = Config()
        config.SEED = seed
        config.SAVE_DIR = os.path.join(config.BASE_SAVE_DIR, f"seed{seed}")
        config.METRICS_FILE = os.path.join(config.SAVE_DIR, "training_metrics.csv")
        config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return config

class AssemblyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class AssemblyTrainer:
    def __init__(self, model, tokenizer, device, config, special_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.special_tokens = special_tokens
        self.best_f1 = 0
        self.early_stopping_counter = 0
        self.start_time = time.time()
        
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        
        # Initialize metrics file
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        """Initialize the metrics CSV file with headers"""
        if not os.path.exists(self.config.METRICS_FILE):
            with open(self.config.METRICS_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'model_name', 'epoch', 
                    'train_loss', 'val_loss', 'train_f1', 'val_f1',
                    'accuracy', 'precision', 'recall', 'f2_score', 'mcc',
                    'tp', 'tn', 'fp', 'fn', 'train_time', 'gpu_mem', 'params'
                ])
    
    def get_gpu_memory(self):
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            return gpu.memoryUsed
        return 0
    
    def count_parameters(self):
        """Count trainable parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        losses = []
        preds = []
        targets = []
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Store metrics
            losses.append(loss.item())
            _, pred = torch.max(outputs.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': np.mean(losses)})

        # Calculate epoch metrics
        train_loss = np.mean(losses)
        train_f1 = f1_score(targets, preds, average='binary')
        return train_loss, train_f1
        
    def evaluate(self, dataloader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        losses = []
        preds = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Store metrics
                losses.append(outputs.loss.item())
                _, pred = torch.max(outputs.logits, dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        loss = np.mean(losses)
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='binary')
        recall = recall_score(targets, preds, average='binary')
        f1 = f1_score(targets, preds, average='binary')
        f2 = (5 * precision * recall) / (4 * precision + recall)
        mcc = matthews_corrcoef(targets, preds)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        
        return loss, f1, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2,
            'mcc': mcc,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def save_metrics(self, model_name, epoch, train_loss, val_loss, train_f1, val_f1, metrics):
        """Save training metrics to CSV"""
        train_time = time.time() - self.start_time
        gpu_mem = self.get_gpu_memory()
        params = self.count_parameters()
        
        with open(self.config.METRICS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                epoch,
                train_loss,
                val_loss,
                train_f1,
                val_f1,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f2_score'],
                metrics['mcc'],
                metrics['tp'],
                metrics['tn'],
                metrics['fp'],
                metrics['fn'],
                train_time,
                gpu_mem,
                params
            ])
    
    def print_metrics(self, metrics, prefix=""):
        """Print evaluation metrics"""
        print(f"\n{prefix} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    
    def save_model(self, model_name, val_f1):
        """Save the best model checkpoint"""
        safe_model_name = model_name.replace('/', '_')
        model_path = os.path.join(self.config.SAVE_DIR, f"best_model_{safe_model_name}.pt")
        tokenizer_path = os.path.join(self.config.SAVE_DIR, f"tokenizer_{safe_model_name}")

        # Save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_name': model_name,
            'special_tokens': self.special_tokens,
            'config': self.config,
            'tokenizer_config': self.tokenizer.init_kwargs
        }, model_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(tokenizer_path)
        
        print(f"\nNew best model saved with Val F1: {val_f1:.4f}")
    
    def train(self, train_loader, val_loader, test_loader, optimizer, scheduler, model_name):
        """Main training loop"""
        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            
            # Train for one epoch
            train_loss, train_f1 = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Evaluate on validation set
            val_loss, val_f1, val_metrics = self.evaluate(val_loader)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            self.print_metrics(val_metrics, "Validation")
            
            # Save metrics
            self.save_metrics(model_name, epoch, train_loss, val_loss, train_f1, val_f1, val_metrics)
            
            # Check for best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.early_stopping_counter = 0
                self.save_model(model_name, val_f1)
                
                # Save test loader for later evaluation
                test_loader_path = os.path.join(self.config.SAVE_DIR, "test_loader.pt")
                torch.save(test_loader, test_loader_path)
                
                print(f"Trainable Parameters: {self.count_parameters():,}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Final evaluation on test set
        print("\nTraining completed. Evaluating on test set...")
        test_loss, test_f1, test_metrics = self.evaluate(test_loader)
        self.print_metrics(test_metrics, "Test")
        
        # Save test metrics
        self._save_test_metrics(test_loss, test_metrics)

    def _save_test_metrics(self, test_loss, test_metrics):
        """Save test metrics and resource usage"""
        total_train_time = time.time() - self.start_time
        gpu_memory = self.get_gpu_memory()
        
        test_metrics_path = os.path.join(self.config.SAVE_DIR, "test_metrics.txt")
        with open(test_metrics_path, 'w') as f:
            f.write("Test Metrics:\n")
            f.write(f"Loss: {test_loss:.4f}\n")
            for key, value in test_metrics.items():
                f.write(f"{key.replace('_', ' ').capitalize()}: {value:.4f}\n" 
                       if isinstance(value, float) 
                       else f"{key.replace('_', ' ').capitalize()}: {value}\n")
            
            f.write("\nResource Usage:\n")
            f.write(f"Total Training Time: {total_train_time:.2f} seconds\n")
            f.write(f"GPU Memory Used: {gpu_memory} MB\n")

def get_stratified_splits(texts, labels, seed, save_dir):
    """Create stratified train/val/test splits"""
    # First split: 90% train+val, 10% test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=seed)
    
    # Second split: 80% train, 10% val (of original)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.111, stratify=train_val_labels, random_state=seed)
    
    # Save test set
    os.makedirs(save_dir, exist_ok=True)
    test_df = pd.DataFrame({'func': test_texts, 'label': test_labels})
    test_df.to_csv(os.path.join(save_dir, "test_set.csv"), index=False)
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def initialize_tokenizer(model_name):
    """Initialize and configure the tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Configure special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add custom special tokens
    special_tokens = ['<hex>', '<reg>', '<mem>', '<imm>', '<label>', '<inst>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    return tokenizer, special_tokens

def initialize_model(model_name, tokenizer, device):
    """Initialize the model with proper configuration"""
    model = GPT2ForSequenceClassification.from_pretrained(
        model_name,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=2  # Binary classification
    ).to(device)
    
    # Resize embeddings for added tokens
    model.resize_token_embeddings(len(tokenizer))
    
    return model

def train_model(seed):
    """Main function to train the model with a specific seed"""
    config = Config.get_config(seed)
    
    # Set random seeds for reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    # Load and prepare data
    df = pd.read_csv(config.DATA_PATH).sample(frac=1, random_state=config.SEED)
    texts = df['func'].values
    labels = df['label'].values
    
    for model_name in config.MODEL_NAMES.values():
        print(f"\n{'='*50}")
        print(f"Training {model_name} with seed {config.SEED}")
        print(f"{'='*50}")
        
        # Initialize tokenizer and model
        tokenizer, special_tokens = initialize_tokenizer(model_name)
        model = initialize_model(model_name, tokenizer, config.DEVICE)
        print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        # Create data splits
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = \
            get_stratified_splits(texts, labels, config.SEED, config.SAVE_DIR)
        
        # Create datasets and dataloaders
        train_dataset = AssemblyDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
        val_dataset = AssemblyDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)
        test_dataset = AssemblyDataset(test_texts, test_labels, tokenizer, config.MAX_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Set up optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        total_steps = len(train_loader) * config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Train the model
        trainer = AssemblyTrainer(model, tokenizer, config.DEVICE, config, special_tokens)
        trainer.train(train_loader, val_loader, test_loader, optimizer, scheduler, model_name)

def main():
    """Run multiple experiments with different seeds"""
    num_runs = 10
    base_seed = Config.BASE_SEED
    
    for run in range(num_runs):
        seed = base_seed + run
        print(f"\n{'#'*60}")
        print(f"Starting run {run+1}/{num_runs} with seed {seed}")
        print(f"{'#'*60}")
        
        train_model(seed)
        
        print(f"\nCompleted run {run+1}/{num_runs} with seed {seed}")
        print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()