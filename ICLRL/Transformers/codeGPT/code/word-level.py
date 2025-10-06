import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm
from torch.optim import AdamW
import csv
from datetime import datetime
import time
import psutil
import GPUtil

class Config:
    MODEL_NAMES = {
        'codegpt': 'microsoft/CodeGPT-small-py'
    }
    
    # Training parameters
    BASE_SEED = 1000  # Base seed to generate other seeds
    BATCH_SIZE = 16
    MAX_LEN = 512
    EPOCHS = 40
    LEARNING_RATE = 2e-5
    EARLY_STOPPING_PATIENCE = 5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    
    # Paths
    DATA_PATH = "E:/models/workB/ICLRL/data/assembly2025.csv"
    BASE_SAVE_DIR = "E:/models/workB/ICLRL/Transformers/codegpt/results-word-level"
    
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

class AssemblyClassifier(nn.Module):
    def __init__(self, model_name, n_classes=2, freeze_bert=False):
        super(AssemblyClassifier, self).__init__()
        
        # Load CodeGPT model (GPT2LMHeadModel is used for CodeGPT)
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Remove the language modeling head since we're doing classification
        self.transformer.lm_head = nn.Identity()
        
        # Get the hidden size from the transformer config
        self.hidden_size = self.transformer.config.n_embd
        
        # BiGRU Layer
        self.bigru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),  # 2*256 for bidirectional
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
        
        if freeze_bert:
            # Freeze token embeddings
            for param in self.transformer.transformer.wte.parameters():
                param.requires_grad = False
            for param in self.transformer.transformer.wpe.parameters():
                param.requires_grad = False
            
            # Freeze first 8 layers of the transformer
            for layer in self.transformer.transformer.h[:8]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Get transformer outputs (hidden states)
        transformer_outputs = self.transformer.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the last hidden state
        word_embeddings = transformer_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # BiGRU processing
        gru_out, _ = self.bigru(word_embeddings)  # [batch, seq_len, 2*hidden_size]
        
        # Attention mechanism
        attention_weights = self.attention(gru_out)  # [batch, seq_len, 1]
        context_vector = torch.sum(attention_weights * gru_out, dim=1)  # [batch, 512]
        
        # Classification
        logits = self.classifier(context_vector)
        return logits, attention_weights

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
        
        if not os.path.exists(config.METRICS_FILE):
            with open(config.METRICS_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'model_name', 'epoch', 
                    'train_loss', 'val_loss', 'train_f1', 'val_f1',
                    'accuracy', 'precision', 'recall', 'f2_score', 'mcc',
                    'tp', 'tn', 'fp', 'fn', 'train_time', 'gpu_mem', 'params'
                ])

    def get_disk_usage(self):
        """Returns disk usage in MB for the save directory"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.config.SAVE_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)  # Convert to MB

    def measure_inference_time(self, dataloader, num_runs=10):
        """Measures average inference time in seconds"""
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    _ = self.model(input_ids, attention_mask)
        return (time.time() - start_time) / num_runs

    def get_gpu_memory(self):
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            return gpu.memoryUsed
        return 0
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        losses = []
        preds = []
        targets = []
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs, attn_weights = self.model(input_ids, attention_mask)
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
                
                outputs, attn_weights = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                
                losses.append(loss.item())
                _, pred = torch.max(outputs, dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        loss = np.mean(losses)
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='binary', zero_division=0)
        recall = recall_score(targets, preds, average='binary', zero_division=0)
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-10)  # Avoid division by zero
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
        print(f"\n{prefix} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"F2 Score: {metrics['f2_score']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    
    def train(self, train_loader, val_loader, test_loader, optimizer, scheduler, model_name):
        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            
            train_loss, train_f1 = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_f1, val_metrics = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            self.print_metrics(val_metrics, "Validation")
            
            self.save_metrics(model_name, epoch, train_loss, val_loss, train_f1, val_f1, val_metrics)
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.early_stopping_counter = 0
                
                safe_model_name = model_name.replace('/', '_')
                model_path = os.path.join(self.config.SAVE_DIR, f"best_model_{safe_model_name}.pt")
                tokenizer_path = os.path.join(self.config.SAVE_DIR, f"tokenizer_{safe_model_name}")

                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                # Save model checkpoint
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'tokenizer_name': model_name,
                    'special_tokens': self.special_tokens,
                    'config': self.config,
                    'tokenizer_config': self.tokenizer.init_kwargs if hasattr(self.tokenizer, 'init_kwargs') else {}
                }, model_path)

                # Save tokenizer in Hugging Face format
                self.tokenizer.save_pretrained(tokenizer_path)

                # Save test loader
                test_loader_path = os.path.join(self.config.SAVE_DIR, "test_loader.pt")
                torch.save(test_loader, test_loader_path)
            
                print(f"Trainable Parameters: {self.count_parameters():,}")
                print(f"\nNew best model saved with Val F1: {val_f1:.4f}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Evaluate on test set after training
        test_loss, test_f1, test_metrics = self.evaluate(test_loader)
        self.print_metrics(test_metrics, "Test")

        # Get resource stats
        total_train_time = time.time() - self.start_time
        gpu_memory = self.get_gpu_memory()
        disk_usage = self.get_disk_usage()

        # Save test metrics and resource usage to a text file
        test_metrics_path = os.path.join(self.config.SAVE_DIR, "test_metrics.txt")
        with open(test_metrics_path, 'w') as f:
            f.write("Test Metrics:\n")
            f.write(f"Loss: {test_loss:.4f}\n")
            for key, value in test_metrics.items():
                f.write(f"{key.replace('_', ' ').capitalize()}: {value:.4f}\n" if isinstance(value, float) else f"{key.replace('_', ' ').capitalize()}: {value}\n")
            
            f.write("\nResource Usage:\n")
            f.write(f"Total Training Time: {total_train_time:.2f} seconds\n")
            f.write(f"GPU Memory Used: {gpu_memory} MB\n")
            f.write(f"Disk Space Used: {disk_usage:.2f} MB\n")

def get_stratified_splits(texts, labels, seed, save_dir):
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=seed)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.111, stratify=train_val_labels, random_state=seed)
    
    # Save test set as CSV
    os.makedirs(save_dir, exist_ok=True)
    test_df = pd.DataFrame({'func': test_texts, 'label': test_labels})
    test_df.to_csv(os.path.join(save_dir, "test_set.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test_set_copy.csv"), index=False)
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def train_model(seed):
    config = Config.get_config(seed)
    
    # Set all seeds
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    
    # Load and shuffle data
    df = pd.read_csv(config.DATA_PATH).sample(frac=1, random_state=config.SEED)
    texts = df['func'].values
    labels = df['label'].values
    
    for model_name in config.MODEL_NAMES.values():
        print(f"\n{'='*50}")
        print(f"Training {model_name} with seed {config.SEED}")
        print(f"{'='*50}")
        
        # Initialize tokenizer and add special tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token for GPT-based models (they don't have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        special_tokens = ['<hex>', '<reg>', '<mem>', '<imm>', '<label>', '<inst>']
        tokenizer.add_tokens(special_tokens)
        
        # Create stratified splits
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = \
            get_stratified_splits(texts, labels, config.SEED, config.SAVE_DIR)
        
        # Create datasets
        train_dataset = AssemblyDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
        val_dataset = AssemblyDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)
        test_dataset = AssemblyDataset(test_texts, test_labels, tokenizer, config.MAX_LEN)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Initialize model
        model = AssemblyClassifier(model_name, freeze_bert=True).to(config.DEVICE)
        model.transformer.resize_token_embeddings(len(tokenizer))
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        total_steps = len(train_loader) * config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.WARMUP_STEPS, num_training_steps=total_steps)
        
        # Train
        trainer = AssemblyTrainer(model, tokenizer, config.DEVICE, config, special_tokens)
        trainer.train(train_loader, val_loader, test_loader, optimizer, scheduler, model_name)

def main():
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