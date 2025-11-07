import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from typing import Dict, List, Tuple, Any
import os
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.utils.class_weight import compute_class_weight
import json
import time

from config import LegalBertConfig
from model import HierarchicalLegalBERT, LegalBertTokenizer
from risk_discovery import UnsupervisedRiskDiscovery, LDARiskDiscovery
from data_loader import CUADDataLoader
from focal_loss import FocalLoss, compute_class_weights
from risk_postprocessing import merge_duplicate_topics, detect_duplicate_topics, validate_cluster_quality

def collate_batch(batch):
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids_batch = []
    attention_mask_batch = []
    risk_labels_batch = []
    severity_scores_batch = []
    importance_scores_batch = []
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        current_len = input_ids.size(0)
        
        if current_len < max_len:
            padding_len = max_len - current_len
            input_ids = torch.cat([input_ids, torch.zeros(padding_len, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)])
        
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        risk_labels_batch.append(item['risk_label'])
        severity_scores_batch.append(item['severity_score'])
        importance_scores_batch.append(item['importance_score'])
    
    return {
        'input_ids': torch.stack(input_ids_batch),
        'attention_mask': torch.stack(attention_mask_batch),
        'risk_label': torch.stack(risk_labels_batch),
        'severity_score': torch.stack(severity_scores_batch),
        'importance_score': torch.stack(importance_scores_batch)
    }

class LegalClauseDataset(Dataset):
    
    def __init__(self, clauses: List[str], risk_labels: List[int], 
                 severity_scores: List[float], importance_scores: List[float],
                 tokenizer: LegalBertTokenizer, max_length: int = 512):
        self.clauses = clauses
        self.risk_labels = risk_labels
        self.severity_scores = severity_scores
        self.importance_scores = importance_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.clauses)
    
    def __getitem__(self, idx):
        clause = self.clauses[idx]
        
        encoded = self.tokenizer.tokenize_clauses([clause], self.max_length)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'risk_label': torch.tensor(self.risk_labels[idx], dtype=torch.long),
            'severity_score': torch.tensor(self.severity_scores[idx], dtype=torch.float),
            'importance_score': torch.tensor(self.importance_scores[idx], dtype=torch.float)
        }

class LegalBertTrainer:
    def __init__(self, config: LegalBertConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        risk_method = config.risk_discovery_method.lower()
        
        if risk_method == 'lda':
            print(f" Using LDA (Topic Modeling) for risk discovery")
            self.risk_discovery = LDARiskDiscovery(
                n_clusters=config.risk_discovery_clusters,
                doc_topic_prior=config.lda_doc_topic_prior,
                topic_word_prior=config.lda_topic_word_prior,
                max_iter=config.lda_max_iter,
                max_features=config.lda_max_features,
                learning_method=config.lda_learning_method,
                random_state=42
            )
        elif risk_method == 'kmeans':
            print(f" Using K-Means for risk discovery")
            self.risk_discovery = UnsupervisedRiskDiscovery(
                n_clusters=config.risk_discovery_clusters,
                random_state=42
            )
        else:
            print(f"  Unknown risk discovery method '{risk_method}', defaulting to LDA")
            self.risk_discovery = LDARiskDiscovery(
                n_clusters=config.risk_discovery_clusters,
                doc_topic_prior=config.lda_doc_topic_prior,
                topic_word_prior=config.lda_topic_word_prior,
                max_iter=config.lda_max_iter,
                max_features=config.lda_max_features,
                learning_method=config.lda_learning_method,
                random_state=42
            )
        
        self.tokenizer = LegalBertTokenizer(config.bert_model_name)
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'per_class_recall': []
        }
        
        if config.use_focal_loss:
            print(" Using Focal Loss for classification (gamma=2.5)")
            self.classification_loss = None
        else:
            print("  Using standard CrossEntropyLoss (not recommended)")
            self.classification_loss = nn.CrossEntropyLoss()
        
        self.regression_loss = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        data_loader = CUADDataLoader(data_path)
        df_clauses, contracts = data_loader.load_data()
        splits = data_loader.create_splits()
        
        train_clauses = splits['train']['clause_text'].tolist()
        
        discovered_patterns = self.risk_discovery.discover_risk_patterns(train_clauses)
        
        print("\n Validating discovered risk patterns...")
        validation_report = validate_cluster_quality(discovered_patterns, min_cluster_size=150)
        
        if not validation_report['is_valid']:
            print("  Cluster quality issues detected:")
            for issue in validation_report['issues']:
                print(f"   - {issue}")
        
        if validation_report['warnings']:
            for warning in validation_report['warnings']:
                print(f"     {warning}")
        
        merge_rules = detect_duplicate_topics(discovered_patterns)
        
        if merge_rules:
            print(f"\n Merging {len(merge_rules)} duplicate topic groups...")
            discovered_patterns, original_labels = merge_duplicate_topics(
                discovered_patterns,
                self.risk_discovery.cluster_labels,
                merge_rules
            )
            self.risk_discovery.discovered_patterns = discovered_patterns
            self.risk_discovery.cluster_labels = original_labels
            self.risk_discovery.n_clusters = len(discovered_patterns)
            print(f" Merged to {self.risk_discovery.n_clusters} distinct risk categories\n")
        
        train_risk_labels = self.risk_discovery.get_risk_labels(train_clauses)
        
        if self.config.use_focal_loss:
            print("\n Computing class weights for Focal Loss...")
            class_weights = compute_class_weights(
                train_risk_labels,
                num_classes=self.risk_discovery.n_clusters,
                minority_boost=self.config.minority_class_boost
            )
            
            self.classification_loss = FocalLoss(
                alpha=class_weights,
                gamma=self.config.focal_loss_gamma,
                reduction='mean'
            )
            print(f" Focal Loss initialized with γ={self.config.focal_loss_gamma}\n")
        
        datasets = {}
        dataloaders = {}
        
        for split_name, split_data in splits.items():
            clauses = split_data['clause_text'].tolist()
            
            risk_labels = self.risk_discovery.get_risk_labels(clauses)
            
            severity_scores = self._generate_synthetic_scores(clauses, 'severity')
            importance_scores = self._generate_synthetic_scores(clauses, 'importance')
            
            dataset = LegalClauseDataset(
                clauses=clauses,
                risk_labels=risk_labels,
                severity_scores=severity_scores,
                importance_scores=importance_scores,
                tokenizer=self.tokenizer,
                max_length=self.config.max_sequence_length
            )
            
            datasets[split_name] = dataset
            
            shuffle = (split_name == 'train')
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=collate_batch
            )
            dataloaders[split_name] = dataloader
        
        print(f" Data preparation complete!")
        print(f" Discovered {len(discovered_patterns)} risk patterns")
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test']
    
    def _generate_synthetic_scores(self, clauses: List[str], score_type: str) -> List[float]:
        scores = []
        
        for clause in clauses:
            features = self.risk_discovery.extract_risk_features(clause)
            
            if score_type == 'severity':
                score = (
                    features.get('risk_intensity', 0) * 30 +
                    features.get('obligation_strength', 0) * 20 +
                    features.get('prohibition_terms_density', 0) * 100 +
                    features.get('liability_terms_density', 0) * 100 +
                    min(features.get('monetary_terms_count', 0) * 0.5, 2)
                )
            else:
                score = (
                    features.get('legal_complexity', 0) * 30 +
                    min(features.get('clause_length', 0) / 50, 1) * 20 +
                    features.get('conditional_risk_density', 0) * 100 +
                    features.get('obligation_terms_complexity', 0) * 100 +
                    features.get('temporal_urgency_density', 0) * 50
                )
            
            normalized_score = min(max(score, 0), 10)
            scores.append(normalized_score)
        
        return scores
    
    def setup_training(self, train_loader: DataLoader):
        num_discovered_risks = self.risk_discovery.n_clusters
        
        print(" Using Hierarchical BERT model (context-aware)")
        self.model = HierarchicalLegalBERT(
            config=self.config,
            num_discovered_risks=num_discovered_risks,
            hidden_dim=self.config.hierarchical_hidden_dim,
            num_lstm_layers=self.config.hierarchical_num_lstm_layers
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        if self.config.use_lr_scheduler:
            total_steps = len(train_loader) * self.config.num_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.scheduler_pct_start,
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=10000.0
            )
            print(f" OneCycleLR scheduler initialized (warmup={self.config.scheduler_pct_start*100:.0f}%)")
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(train_loader) * self.config.num_epochs
            )
            print("  Using basic CosineAnnealingLR (not recommended)")
        
        print(f" Model initialized with {num_discovered_risks} discovered risk categories")
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        classification_loss = self.classification_loss(
            outputs['risk_logits'],
            batch['risk_label']
        )
        
        severity_loss = self.regression_loss(
            outputs['severity_score'],
            batch['severity_score']
        )
        
        importance_loss = self.regression_loss(
            outputs['importance_score'],
            batch['importance_score']
        )
        
        total_loss = (
            self.config.task_weights['classification'] * classification_loss +
            self.config.task_weights['severity'] * severity_loss +
            self.config.task_weights['importance'] * importance_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'severity_loss': severity_loss,
            'importance_loss': importance_loss
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        loss_components = {'classification': 0, 'severity': 0, 'importance': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            risk_labels = batch['risk_label'].to(self.device)
            severity_scores = batch['severity_score'].to(self.device)
            importance_scores = batch['importance_score'].to(self.device)
            
            outputs = self.model.forward_single_clause(input_ids, attention_mask)
            
            batch_for_loss = {
                'risk_label': risk_labels,
                'severity_score': severity_scores,
                'importance_score': importance_scores
            }
            
            losses = self.compute_loss(outputs, batch_for_loss)
            
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += losses['total_loss'].item()
            
            predictions = torch.argmax(outputs['risk_logits'], dim=-1)
            correct_predictions += (predictions == risk_labels).sum().item()
            total_samples += risk_labels.size(0)
            
            loss_components['classification'] += losses['classification_loss'].item()
            loss_components['severity'] += losses['severity_loss'].item()
            loss_components['importance'] += losses['importance_loss'].item()
            
            if batch_idx % 50 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {losses['total_loss'].item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        return avg_loss, accuracy, loss_components
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray]:
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                risk_labels = batch['risk_label'].to(self.device)
                severity_scores = batch['severity_score'].to(self.device)
                importance_scores = batch['importance_score'].to(self.device)
                
                outputs = self.model.forward_single_clause(input_ids, attention_mask)
                
                batch_for_loss = {
                    'risk_label': risk_labels,
                    'severity_score': severity_scores,
                    'importance_score': importance_scores
                }
                
                losses = self.compute_loss(outputs, batch_for_loss)
                total_loss += losses['total_loss'].item()
                
                predictions = torch.argmax(outputs['risk_logits'], dim=-1)
                correct_predictions += (predictions == risk_labels).sum().item()
                total_samples += risk_labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(risk_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        per_class_recall = recall_score(
            all_labels, 
            all_predictions, 
            average=None,
            zero_division=0
        )
        
        return avg_loss, accuracy, per_class_recall
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        print(f" Starting Legal-BERT training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        self.setup_training(train_loader)
        
        total_start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\n Epoch {epoch+1}/{self.config.num_epochs}")
            
            epoch_start_time = time.time()
            
            train_loss, train_acc, loss_components = self.train_epoch(train_loader, epoch)
            
            val_loss, val_acc, per_class_recall = self.validate_epoch(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['per_class_recall'].append(per_class_recall.tolist())
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Loss Components - Class: {loss_components['classification']:.4f}, "
                  f"Sev: {loss_components['severity']:.4f}, Imp: {loss_components['importance']:.4f}")
            
            print(f"  Per-Class Recall:")
            critical_classes = [0, 5]
            for cls_idx, recall in enumerate(per_class_recall):
                marker = "  CRITICAL" if cls_idx in critical_classes else ""
                print(f"    Class {cls_idx}: {recall:.3f}{marker}")
            
            print(f"    Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"   New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"    No improvement ({self.patience_counter}/{self.config.early_stopping_patience})")
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n Early stopping triggered after {epoch+1} epochs")
                    break
            
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"   Loss Components:")
            print(f"    Classification: {loss_components['classification']:.4f}")
            print(f"    Severity: {loss_components['severity']:.4f}")
            print(f"    Importance: {loss_components['importance']:.4f}")
            print(f"    Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)")
            
            self.save_checkpoint(epoch)
        
        total_time = time.time() - total_start_time
        
        print(f"\n Training complete!")
        print(f"  Total Training Time: {total_time:.2f}s ({total_time/60:.2f} minutes / {total_time/3600:.2f} hours)")
        print(f"  Average Time per Epoch: {total_time/self.config.num_epochs:.2f}s")
        
        return self.training_history
    
    def save_checkpoint(self, epoch: int):
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'discovered_patterns': self.risk_discovery.discovered_patterns
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'legal_bert_epoch_{epoch+1}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        print(f" Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        num_discovered_risks = len(checkpoint['discovered_patterns'])
        self.model = HierarchicalLegalBERT(
            config=checkpoint['config'],
            num_discovered_risks=num_discovered_risks,
            hidden_dim=checkpoint['config'].hierarchical_hidden_dim,
            num_lstm_layers=checkpoint['config'].hierarchical_num_lstm_layers
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.training_history = checkpoint['training_history']
        self.risk_discovery.discovered_patterns = checkpoint['discovered_patterns']
        
        print(f" Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint['epoch']