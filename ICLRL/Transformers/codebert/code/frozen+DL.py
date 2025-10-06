import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, matthews_corrcoef, confusion_matrix)
from tqdm import tqdm
from torch.optim import AdamW
import csv
from collections import defaultdict
from datetime import datetime
import time
import psutil
import GPUtil
import warnings
warnings.filterwarnings('ignore')

class Config:
    MODEL_NAMES = {
        'codebert': 'google-bert/bert-base-uncased'
        #google-bert/bert-base-uncased
        #microsoft/codebert-base
    }
    
    # Training parameters - optimized for frozen feature extraction
    SEEDS = [42, 123, 456, 789, 1011]  # Reduced for faster experimentation
    BATCH_SIZE = 32  # Increased since we're not training transformer layers
    MAX_LEN = 512
    EPOCHS = 30  # Reduced since we're mainly training classifier heads
    LEARNING_RATE = 1e-3  # Higher LR for classifier training
    EARLY_STOPPING_PATIENCE = 6
    WEIGHT_DECAY = 1e-4
    WARMUP_STEPS = 50
    
    # Feature extraction parameters
    FEATURE_DIM = 768  # CodeBERT hidden size
    DROPOUT_RATE = 0.3
    HIDDEN_DIMS = [512, 256, 128]  # Multi-layer classifier
    
    # Base paths
    BASE_DATA_PATH = "E:/models/workB/ICLRL/data/assembly2025.csv"
    BASE_SAVE_DIR = "E:/models/workB/ICLRL/Transformers/codebert/bertResulsFrozen"
    
    @classmethod
    def get_device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
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

class CodeBERTFeatureExtractor(nn.Module):
    """Frozen CodeBERT feature extractor with enhanced classifier head"""
    
    def __init__(self, model_name, n_classes=2, config=None):
        super(CodeBERTFeatureExtractor, self).__init__()
        self.config = config or Config()
        
        # Load and freeze CodeBERT
        self.codebert = AutoModel.from_pretrained(model_name)
        for param in self.codebert.parameters():
            param.requires_grad = False
        
        # Set to evaluation mode to disable dropout in CodeBERT
        self.codebert.eval()
        
        # Enhanced classifier head with multiple layers
        layers = []
        input_dim = self.codebert.config.hidden_size
        
        for hidden_dim in self.config.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.DROPOUT_RATE)
            ])
            input_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(input_dim, n_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Feature aggregation strategies
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.codebert.config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable pooling weights
        self.pooling_weights = nn.Parameter(torch.randn(4))  # For different pooling strategies
        
    def forward(self, input_ids, attention_mask, return_features=False):
        # Extract features with frozen CodeBERT
        with torch.no_grad():
            outputs = self.codebert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Multiple feature extraction strategies
        last_hidden_states = outputs.last_hidden_state
        
        # 1. CLS token
        cls_output = last_hidden_states[:, 0, :]
        
        # 2. Mean pooling
        mean_output = torch.mean(last_hidden_states, dim=1)
        
        # 3. Max pooling
        max_output = torch.max(last_hidden_states, dim=1)[0]
        
        # 4. Attention pooling
        attended_output, _ = self.attention_pooling(
            last_hidden_states, last_hidden_states, last_hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        attended_output = attended_output.mean(dim=1)
        
        # Weighted combination of pooling strategies
        pooling_weights = F.softmax(self.pooling_weights, dim=0)
        combined_features = (
            pooling_weights[0] * cls_output +
            pooling_weights[1] * mean_output +
            pooling_weights[2] * max_output +
            pooling_weights[3] * attended_output
        )
        
        if return_features:
            return combined_features
        
        # Classification
        logits = self.classifier(combined_features)
        return logits

class AssemblyTrainer:
    def __init__(self, model, tokenizer, device, config, special_tokens, seed):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.special_tokens = special_tokens
        self.seed = seed
        self.best_f1 = 0
        self.early_stopping_counter = 0
        self.training_start_time = None
        self.total_training_time = 0
        
        # Create seed-specific directories
        self.seed_dir = os.path.join(config.SAVE_DIR, f"seed{seed}")
        os.makedirs(self.seed_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.seed_dir, "training_metrics.csv")
        
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'model_name', 'epoch', 'approach',
                    'train_loss', 'val_loss', 'train_f1', 'val_f1',
                    'accuracy', 'precision', 'recall', 'f2_score', 'mcc',
                    'tp', 'tn', 'fp', 'fn', 'train_time', 'gpu_mem_mb', 'params'
                ])

    def calculate_metrics(self, targets, preds):
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='binary')
        recall = recall_score(targets, preds, average='binary')
        f1 = f1_score(targets, preds, average='binary')
        f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
        mcc = matthews_corrcoef(targets, preds)
        
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        
        return {
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

    def get_gpu_memory_usage(self):
        """Get GPU memory usage in MB (similar to Windows Performance Monitor)"""
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Get the first GPU (or you can specify which one)
                    gpu = gpus[0]
                    return gpu.memoryUsed  # Returns memory in MB
                else:
                    return 0
            except:
                # Fallback to PyTorch memory tracking
                return torch.cuda.memory_allocated() / 1024 / 1024  # Convert bytes to MB
        return 0
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        # Keep CodeBERT in eval mode
        self.model.codebert.eval()
        
        losses = []
        preds = []
        targets = []
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            _, pred = torch.max(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': np.mean(losses)})

        train_loss = np.mean(losses)
        train_f1 = f1_score(targets, preds, average='binary')
        return train_loss, train_f1
        
    def evaluate(self, dataloader):
        self.model.eval()
        losses = []
        preds = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                losses.append(loss.item())
                _, pred = torch.max(outputs, dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        loss = np.mean(losses)
        metrics = self.calculate_metrics(targets, preds)
        
        return loss, metrics['f1_score'], metrics
    
    def save_metrics(self, model_name, epoch, approach, train_loss, val_loss, train_f1, val_f1, metrics):
        current_time = time.time() - self.training_start_time if self.training_start_time else 0
        gpu_mem = self.get_gpu_memory_usage()
        params = self.count_parameters()
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                epoch,
                approach,
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
                current_time,
                gpu_mem,
                params
            ])
    
    def save_test_metrics_to_file(self, test_metrics, model_name):
        """Save test metrics to a text file"""
        test_metrics_file = os.path.join(self.seed_dir, "test_metrics.txt")
        
        with open(test_metrics_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TEST SET EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Training Time: {self.total_training_time:.2f} seconds\n")
            f.write(f"GPU Memory Used: {self.get_gpu_memory_usage():.2f} MB\n")
            f.write(f"Trainable Parameters: {self.count_parameters():,}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("CLASSIFICATION METRICS\n")
            f.write("="*60 + "\n")
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {test_metrics['f1_score']:.4f}\n")
            f.write(f"F2 Score: {test_metrics['f2_score']:.4f}\n")
            f.write(f"Matthews Correlation Coefficient: {test_metrics['mcc']:.4f}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("="*60 + "\n")
            f.write(f"True Positives (TP): {test_metrics['tp']}\n")
            f.write(f"True Negatives (TN): {test_metrics['tn']}\n")
            f.write(f"False Positives (FP): {test_metrics['fp']}\n")
            f.write(f"False Negatives (FN): {test_metrics['fn']}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Best Validation F1: {self.best_f1:.4f}\n")
            f.write(f"Total Training Time: {self.total_training_time:.2f} seconds\n")
            f.write(f"Final GPU Memory Usage: {self.get_gpu_memory_usage():.2f} MB\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def print_metrics(self, metrics, prefix=""):
        print(f"\n{prefix} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    
    def train(self, train_loader, val_loader, test_loader, optimizer, scheduler, model_name):
        print(f"\n{'='*60}")
        print(f"Training Deep Learning Approach (Frozen CodeBERT + Neural Classifier)")
        print(f"{'='*60}")
        
        # Record training start time
        self.training_start_time = time.time()
        
        # Train DL approach
        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            
            train_loss, train_f1 = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_f1, val_metrics = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            print(f"GPU Memory: {self.get_gpu_memory_usage():.2f} MB")
            self.print_metrics(val_metrics, "Validation")
            
            self.save_metrics(model_name, epoch, "DL", train_loss, val_loss, train_f1, val_f1, val_metrics)
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.early_stopping_counter = 0
                
                # Save best DL model
                safe_model_name = model_name.replace('/', '_')
                model_path = os.path.join(self.seed_dir, f"best_dl_model_{safe_model_name}.pt")
                tokenizer_path = os.path.join(self.seed_dir, f"tokenizer_{safe_model_name}")

                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'tokenizer_name': model_name,
                    'special_tokens': self.special_tokens,
                    'config': self.config,
                    'seed': self.seed,
                    'tokenizer_config': self.tokenizer.init_kwargs
                }, model_path)

                self.tokenizer.save_pretrained(tokenizer_path)
                print(f"New best DL model saved with Val F1: {val_f1:.4f}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Calculate total training time
        self.total_training_time = time.time() - self.training_start_time
        
        # Final DL evaluation
        test_loss, test_f1, test_metrics = self.evaluate(test_loader)
        self.print_metrics(test_metrics, "Test")
        
        # Save test metrics to file
        self.save_test_metrics_to_file(test_metrics, model_name)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Best Validation F1: {self.best_f1:.4f}")
        print(f"Test F1: {test_metrics['f1_score']:.4f}")
        print(f"Total Training Time: {self.total_training_time:.2f} seconds")
        print(f"GPU Memory Used: {self.get_gpu_memory_usage():.2f} MB")
        print(f"Trainable Parameters: {self.count_parameters():,}")
        
        return test_metrics

def get_stratified_splits(texts, labels, seed):
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=seed)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.111, stratify=train_val_labels, random_state=seed)
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def run_experiment(seed, config):
    print(f"\n{'='*50}")
    print(f"Starting experiment with seed: {seed}")
    print(f"{'='*50}")
    
    # Update config with current seed's paths
    config.SEED = seed
    config.SAVE_DIR = os.path.join(config.BASE_SAVE_DIR, f"seed{seed}")
    config.METRICS_FILE = os.path.join(config.SAVE_DIR, "training_metrics.csv")
    
    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Load and shuffle data
    df = pd.read_csv(config.BASE_DATA_PATH).sample(frac=1, random_state=seed)
    texts = df['func'].values
    labels = df['label'].values
    
    for model_name in config.MODEL_NAMES.values():
        print(f"\n{'='*50}")
        print(f"Training {model_name} with seed {seed}")
        print(f"{'='*50}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens = ['<hex>', '<reg>', '<mem>', '<imm>', '<label>', '<inst>']
        tokenizer.add_tokens(special_tokens)
        
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = \
            get_stratified_splits(texts, labels, seed)
        
        # Save test set
        test_df = pd.DataFrame({'func': test_texts, 'label': test_labels})
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        test_df.to_csv(os.path.join(config.SAVE_DIR, "test_set.csv"), index=False)
        
        train_dataset = AssemblyDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
        val_dataset = AssemblyDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)
        test_dataset = AssemblyDataset(test_texts, test_labels, tokenizer, config.MAX_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = CodeBERTFeatureExtractor(model_name, config=config).to(config.get_device())
        model.codebert.resize_token_embeddings(len(tokenizer))
        
        # Only optimize classifier parameters
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        
        total_steps = len(train_loader) * config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=config.WARMUP_STEPS, 
            num_training_steps=total_steps
        )
        
        trainer = AssemblyTrainer(model, tokenizer, config.get_device(), config, special_tokens, seed)
        trainer.train(train_loader, val_loader, test_loader, optimizer, scheduler, model_name)

def main():
    config = Config()
    
    # Create base directory if it doesn't exist
    os.makedirs(config.BASE_SAVE_DIR, exist_ok=True)
    
    # Run experiments for each seed sequentially
    for seed in config.SEEDS:
        try:
            run_experiment(seed, config)
            print(f"\n{'='*50}")
            print(f"Completed experiment with seed {seed}")
            print(f"{'='*50}")
        except Exception as e:
            print(f"\n{'!'*50}")
            print(f"Error in experiment with seed {seed}: {str(e)}")
            print(f"{'!'*50}")
            continue

if __name__ == "__main__":
    main()