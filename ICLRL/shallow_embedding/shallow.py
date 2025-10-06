import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
from torch.optim import AdamW
import csv
from collections import defaultdict
from datetime import datetime
import time
import psutil
import GPUtil
import warnings
import re
from typing import List, Dict, Tuple
warnings.filterwarnings('ignore')

class Config:
    """Configuration class for experiment parameters"""
    EMBEDDING_METHODS = ['word2vec', 'fasttext', 'tfidf']
    EMBEDDING_METHOD = 'word2vec'  # Default embedding method
    SEEDS = [42, 123, 456, 789, 1011]
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 8
    WEIGHT_DECAY = 1e-4
    EMBEDDING_DIM = 300
    WINDOW_SIZE = 5
    MIN_COUNT = 2
    WORKERS = 4
    TFIDF_MAX_FEATURES = 10000
    TFIDF_NGRAM_RANGE = (1, 3)
    DROPOUT_RATE = 0.3
    HIDDEN_DIMS = [512, 256, 128]
    BASE_DATA_PATH = "E:/models/workB/ICLRL/data/assembly2025.csv"
    BASE_SAVE_DIR = "E:/models/workB/ICLRL/shallow_embedding"
    
    @classmethod
    def get_device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextPreprocessor:
    """Handles preprocessing and tokenization of assembly code"""
    def __init__(self):
        self.special_tokens = ['<hex>', '<reg>', '<mem>', '<imm>', '<label>', '<inst>']
    
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s<>_,\[\]\+\-\*]', ' ', text)
        return text.strip()
    
    def tokenize_text(self, text: str) -> List[str]:
        return self.preprocess_text(text).split()

class EmbeddingTrainer:
    """Trains shallow embedding models"""
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = TextPreprocessor()
    
    def prepare_corpus(self, texts: List[str]) -> List[List[str]]:
        return [self.preprocessor.tokenize_text(text) for text in texts]
    
    def train_word2vec(self, corpus: List[List[str]], save_path: str) -> Word2Vec:
        print("Training Word2Vec model...")
        model = Word2Vec(
            sentences=corpus,
            vector_size=self.config.EMBEDDING_DIM,
            window=self.config.WINDOW_SIZE,
            min_count=self.config.MIN_COUNT,
            workers=self.config.WORKERS,
            epochs=10,
            sg=1
        )
        model.save(save_path)
        return model
    
    def train_fasttext(self, corpus: List[List[str]], save_path: str) -> FastText:
        print("Training FastText model...")
        model = FastText(
            sentences=corpus,
            vector_size=self.config.EMBEDDING_DIM,
            window=self.config.WINDOW_SIZE,
            min_count=self.config.MIN_COUNT,
            workers=self.config.WORKERS,
            epochs=10,
            sg=1
        )
        model.save(save_path)
        return model
    
    def train_tfidf(self, texts: List[str], save_path: str) -> TfidfVectorizer:
        print("Training TF-IDF vectorizer...")
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        vectorizer = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            min_df=2,
            max_df=0.95
        )
        vectorizer.fit(processed_texts)
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        return vectorizer

class FeatureExtractor:
    """Extracts features using trained embedding models"""
    def __init__(self, embedding_method: str, model, config: Config):
        self.embedding_method = embedding_method
        self.model = model
        self.config = config
        self.preprocessor = TextPreprocessor()
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        if self.embedding_method == 'word2vec':
            return self._extract_word2vec_features(texts)
        elif self.embedding_method == 'fasttext':
            return self._extract_fasttext_features(texts)
        elif self.embedding_method == 'tfidf':
            return self._extract_tfidf_features(texts)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
    
    def _extract_word2vec_features(self, texts: List[str]) -> np.ndarray:
        features = []
        for text in texts:
            tokens = self.preprocessor.tokenize_text(text)
            vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
            feature_vector = np.mean(vectors, axis=0) if vectors else np.zeros(self.config.EMBEDDING_DIM)
            features.append(feature_vector)
        return np.array(features)
    
    def _extract_fasttext_features(self, texts: List[str]) -> np.ndarray:
        features = []
        for text in texts:
            tokens = self.preprocessor.tokenize_text(text)
            vectors = [self.model.wv[token] for token in tokens]
            feature_vector = np.mean(vectors, axis=0) if vectors else np.zeros(self.config.EMBEDDING_DIM)
            features.append(feature_vector)
        return np.array(features)
    
    def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        return self.model.transform(processed_texts).toarray()

class AssemblyDataset(Dataset):
    """Dataset for pre-computed features"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }

class ShallowEmbeddingClassifier(nn.Module):
    """Neural classifier for shallow embeddings"""
    def __init__(self, input_dim: int, n_classes: int = 2, config: Config = None):
        super(ShallowEmbeddingClassifier, self).__init__()
        self.config = config or Config()
        
        layers = []
        current_dim = input_dim
        for hidden_dim in self.config.HIDDEN_DIMS:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.DROPOUT_RATE)
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, n_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features):
        return self.classifier(features)

class AssemblyTrainer:
    """Handles training and evaluation of the classifier"""
    def __init__(self, model, device, config, embedding_method, seed):
        self.model = model
        self.device = device
        self.config = config
        self.embedding_method = embedding_method
        self.seed = seed
        self.best_f1 = 0
        self.early_stopping_counter = 0
        self.training_start_time = None
        self.total_training_time = 0
        
        self.seed_dir = os.path.join(config.BASE_SAVE_DIR, embedding_method, f"seed{seed}")
        os.makedirs(self.seed_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.seed_dir, "training_metrics.csv")
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'embedding_method', 'epoch', 'approach',
                'train_loss', 'val_loss', 'train_f1', 'val_f1',
                'accuracy', 'precision', 'recall', 'f2_score', 'mcc',
                'tp', 'tn', 'fp', 'fn', 'train_time', 'gpu_mem_mb', 'params'
            ])
    
    def calculate_metrics(self, targets, preds):
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
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                return gpus[0].memoryUsed if gpus else 0
            except:
                return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        losses = []
        preds = []
        targets = []
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            optimizer.zero_grad()
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(features)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            _, pred = torch.max(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
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
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(features)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                losses.append(loss.item())
                _, pred = torch.max(outputs, dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        loss = np.mean(losses)
        metrics = self.calculate_metrics(targets, preds)
        return loss, metrics['f1_score'], metrics
    
    def save_metrics(self, epoch, train_loss, val_loss, train_f1, val_f1, metrics):
        current_time = time.time() - self.training_start_time if self.training_start_time else 0
        gpu_mem = self.get_gpu_memory_usage()
        params = self.count_parameters()
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.embedding_method,
                epoch,
                'shallow',
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
    
    def save_test_metrics(self, test_metrics):
        test_metrics_file = os.path.join(self.seed_dir, "test_metrics.txt")
        with open(test_metrics_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"TEST SET EVALUATION RESULTS ({self.embedding_method.upper()})\n")
            f.write("="*60 + "\n")
            f.write(f"Embedding Method: {self.embedding_method}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Training Time: {self.total_training_time:.2f} seconds\n")
            f.write(f"GPU Memory Used: {self.get_gpu_memory_usage():.2f} MB\n")
            f.write(f"Trainable Parameters: {self.count_parameters():,}\n")
            f.write("\n" + "="*60 + "\n")
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {test_metrics['f1_score']:.4f}\n")
            f.write(f"F2 Score: {test_metrics['f2_score']:.4f}\n")
            f.write(f"MCC: {test_metrics['mcc']:.4f}\n")
            f.write(f"TP: {test_metrics['tp']} | TN: {test_metrics['tn']} | FP: {test_metrics['fp']} | FN: {test_metrics['fn']}\n")
    
    def train(self, train_loader, val_loader, test_loader, optimizer):
        print(f"\n{'='*60}")
        print(f"Training {self.embedding_method.upper()} Classifier")
        print(f"{'='*60}")
        
        self.training_start_time = time.time()
        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            train_loss, train_f1 = self.train_epoch(train_loader, optimizer)
            val_loss, val_f1, val_metrics = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            print(f"GPU Memory: {self.get_gpu_memory_usage():.2f} MB")
            
            self.save_metrics(epoch, train_loss, val_loss, train_f1, val_f1, val_metrics)
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.early_stopping_counter = 0
                model_path = os.path.join(self.seed_dir, f"best_model_{self.embedding_method}.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'embedding_method': self.embedding_method,
                    'config': self.config,
                    'seed': self.seed
                }, model_path)
                print(f"New best model saved with Val F1: {val_f1:.4f}")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        self.total_training_time = time.time() - self.training_start_time
        test_loss, test_f1, test_metrics = self.evaluate(test_loader)
        self.save_test_metrics(test_metrics)
        
        print(f"\n{'='*60}")
        print(f"Best Validation F1: {self.best_f1:.4f}")
        print(f"Test F1: {test_metrics['f1_score']:.4f}")
        print(f"Total Training Time: {self.total_training_time:.2f} seconds")
        
        return test_metrics

class CrossSeedEvaluator:
    """Evaluates performance across multiple seeds"""
    def __init__(self, base_save_dir: str, seeds: List[int], embedding_method: str):
        self.base_save_dir = base_save_dir
        self.seeds = seeds
        self.embedding_method = embedding_method
        self.results = defaultdict(list)
    
    def collect_results(self, seed, test_metrics, training_time):
        self.results['seed'].append(seed)
        self.results['accuracy'].append(test_metrics['accuracy'])
        self.results['precision'].append(test_metrics['precision'])
        self.results['recall'].append(test_metrics['recall'])
        self.results['f1_score'].append(test_metrics['f1_score'])
        self.results['f2_score'].append(test_metrics['f2_score'])
        self.results['mcc'].append(test_metrics['mcc'])
        self.results['tp'].append(test_metrics['tp'])
        self.results['tn'].append(test_metrics['tn'])
        self.results['fp'].append(test_metrics['fp'])
        self.results['fn'].append(test_metrics['fn'])
        self.results['training_time'].append(training_time)
    
    def calculate_statistics(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'f2_score', 'mcc', 'training_time']
        stats = {}
        for metric in metrics:
            values = np.array(self.results[metric])
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return stats
    
    def save_report(self):
        stats = self.calculate_statistics()
        results_df = pd.DataFrame(self.results)
        results_file = os.path.join(self.base_save_dir, self.embedding_method, f"cross_seed_results_{self.embedding_method}.csv")
        results_df.to_csv(results_file, index=False)
        
        report_file = os.path.join(self.base_save_dir, self.embedding_method, f"cross_seed_summary_{self.embedding_method}.txt")
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CROSS-SEED EVALUATION SUMMARY ({self.embedding_method.upper()})\n")
            f.write("="*80 + "\n")
            f.write(f"Number of Seeds: {len(self.seeds)}\n")
            f.write(f"Seeds Used: {self.seeds}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nPerformance Statistics (Mean ± Std):\n")
            for metric, stat in stats.items():
                if metric == 'training_time':
                    f.write(f"{metric.title()}: {stat['mean']:.2f} ± {stat['std']:.2f} seconds\n")
                else:
                    f.write(f"{metric.title()}: {stat['mean']:.4f} ± {stat['std']:.4f}\n")
    
    def print_summary(self):
        stats = self.calculate_statistics()
        print(f"\n{'='*80}")
        print(f"CROSS-SEED EVALUATION SUMMARY: {self.embedding_method.upper()}")
        print(f"{'='*80}")
        print(f"Number of Seeds: {len(self.seeds)}")
        print(f"Seeds Used: {self.seeds}")
        print(f"\nPerformance Statistics (Mean ± Std):")
        for metric, stat in stats.items():
            if metric == 'training_time':
                print(f"{metric.title()}: {stat['mean']:.2f} ± {stat['std']:.2f} seconds")
            else:
                print(f"{metric.title()}: {stat['mean']:.4f} ± {stat['std']:.4f}")

def run_experiment(seed: int, config: Config, evaluator: CrossSeedEvaluator) -> Dict:
    config.SAVE_DIR = os.path.join(config.BASE_SAVE_DIR, config.EMBEDDING_METHOD, f"seed{seed}")
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    df = pd.read_csv(config.BASE_DATA_PATH).sample(frac=1, random_state=seed)
    texts = df['func'].values
    labels = df['label'].values
    
    # Split data
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=seed)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.111, stratify=train_val_labels, random_state=seed)
    
    # Save test set
    pd.DataFrame({'func': test_texts, 'label': test_labels}).to_csv(
        os.path.join(config.SAVE_DIR, "test_set.csv"), index=False)
    
    # Train embeddings
    embedding_trainer = EmbeddingTrainer(config)
    corpus = embedding_trainer.prepare_corpus(train_texts)
    embedding_model_path = os.path.join(config.SAVE_DIR, f"{config.EMBEDDING_METHOD}_model")
    
    if config.EMBEDDING_METHOD == 'word2vec':
        embedding_model = embedding_trainer.train_word2vec(corpus, embedding_model_path)
    elif config.EMBEDDING_METHOD == 'fasttext':
        embedding_model = embedding_trainer.train_fasttext(corpus, embedding_model_path)
    elif config.EMBEDDING_METHOD == 'tfidf':
        embedding_model = embedding_trainer.train_tfidf(train_texts, embedding_model_path)
    else:
        raise ValueError(f"Unsupported embedding method: {config.EMBEDDING_METHOD}")
    
    # Extract features
    feature_extractor = FeatureExtractor(config.EMBEDDING_METHOD, embedding_model, config)
    train_features = feature_extractor.extract_features(train_texts)
    val_features = feature_extractor.extract_features(val_texts)
    test_features = feature_extractor.extract_features(test_texts)
    
    # Create datasets and loaders
    train_dataset = AssemblyDataset(train_features, train_labels)
    val_dataset = AssemblyDataset(val_features, val_labels)
    test_dataset = AssemblyDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.WORKERS)
    
    # Initialize model
    input_dim = config.TFIDF_MAX_FEATURES if config.EMBEDDING_METHOD == 'tfidf' else config.EMBEDDING_DIM
    model = ShallowEmbeddingClassifier(input_dim, config=config).to(config.get_device())
    
    # Train
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    trainer = AssemblyTrainer(model, config.get_device(), config, config.EMBEDDING_METHOD, seed)
    test_metrics = trainer.train(train_loader, val_loader, test_loader, optimizer)
    
    evaluator.collect_results(seed, test_metrics, trainer.total_training_time)
    return test_metrics

def main():
    config = Config()
    os.makedirs(config.BASE_SAVE_DIR, exist_ok=True)
    
    for embedding_method in config.EMBEDDING_METHODS:
        config.EMBEDDING_METHOD = embedding_method
        evaluator = CrossSeedEvaluator(config.BASE_SAVE_DIR, config.SEEDS, embedding_method)
        
        for seed in config.SEEDS:
            try:
                print(f"\n{'='*50}")
                print(f"Experiment: Seed {seed}, Embedding {embedding_method.upper()}")
                test_metrics = run_experiment(seed, config, evaluator)
                print(f"Completed: Seed {seed}, Test F1: {test_metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"Error in seed {seed}, {embedding_method}: {str(e)}")
                continue
        
        evaluator.print_summary()
        evaluator.save_report()
        print(f"\nReport saved for {embedding_method.upper()}")

if __name__ == "__main__":
    main()