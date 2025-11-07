import torch
import os
import json
import numpy as np
from datetime import datetime

from config import LegalBertConfig
from trainer import LegalBertTrainer, LegalClauseDataset, collate_batch
from data_loader import CUADDataLoader
from model import HierarchicalLegalBERT
from torch.utils.data import DataLoader

class CalibrationFramework:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.temperature = 1.0
        
    def collect_logits_and_labels(self, data_loader):
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['risk_label']
                
                outputs = self.model.forward_single_clause(input_ids, attention_mask)
                logits = outputs['risk_logits']
                
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        return torch.cat(all_logits), torch.cat(all_labels)
    
    def temperature_scaling(self, val_loader, lr=0.01, max_iter=50):
        print("  Applying temperature scaling...")
        
        logits, labels = self.collect_logits_and_labels(val_loader)
        
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        print(f"   Optimal temperature: {self.temperature:.4f}")
        
        return self.temperature
    
    def apply_temperature(self, logits):
        return logits / self.temperature
    
    def calculate_ece(self, data_loader, n_bins=15):
        print(" Calculating Expected Calibration Error (ECE)...")
        
        confidences = []
        predictions = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['risk_label']
                
                outputs = self.model.forward_single_clause(input_ids, attention_mask)
                logits = self.apply_temperature(outputs['risk_logits'])
                
                probs = torch.softmax(logits, dim=-1)
                conf, pred = torch.max(probs, dim=-1)
                
                confidences.extend(conf.cpu().numpy())
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(predictions[in_bin] == true_labels[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        print(f"  ECE: {ece:.4f}")
        return ece
    
    def calculate_mce(self, data_loader, n_bins=15):
        print(" Calculating Maximum Calibration Error (MCE)...")
        
        confidences = []
        predictions = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['risk_label']
                
                outputs = self.model.forward_single_clause(input_ids, attention_mask)
                logits = self.apply_temperature(outputs['risk_logits'])
                
                probs = torch.softmax(logits, dim=-1)
                conf, pred = torch.max(probs, dim=-1)
                
                confidences.extend(conf.cpu().numpy())
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        mce = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(predictions[in_bin] == true_labels[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        print(f"  MCE: {mce:.4f}")
        return mce

def main():
    print("=" * 80)
    print("  LEGAL-BERT CALIBRATION PIPELINE")
    print("=" * 80)
    
    config = LegalBertConfig()
    
    print("\n Loading trained model...")
    model_path = os.path.join(config.model_save_path, 'final_model.pt')
    
    if not os.path.exists(model_path):
        print(f" Error: Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        hidden_dim = saved_config.hierarchical_hidden_dim
        num_lstm_layers = saved_config.hierarchical_num_lstm_layers
        print(f"   Using saved architecture: hidden_dim={hidden_dim}, lstm_layers={num_lstm_layers}")
    else:
        hidden_dim = config.hierarchical_hidden_dim
        num_lstm_layers = config.hierarchical_num_lstm_layers
        print(f"     Warning: No config in checkpoint, using current config")
    
    print(" Loading Hierarchical BERT model")
    model = HierarchicalLegalBERT(
        config=config,
        num_discovered_risks=len(checkpoint['discovered_patterns']),
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers
    ).to(config.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(" Model loaded successfully!")
    
    print("\n Loading data...")
    data_loader = CUADDataLoader(config.data_path)
    df_clauses, contracts = data_loader.load_data()
    splits = data_loader.create_splits()
    
    trainer = LegalBertTrainer(config)
    
    if 'risk_discovery_model' in checkpoint:
        trainer.risk_discovery = checkpoint['risk_discovery_model']
    else:
        trainer.risk_discovery.discovered_patterns = checkpoint['discovered_patterns']
        trainer.risk_discovery.n_clusters = len(checkpoint['discovered_patterns'])
    
    trainer.model = model
    
    val_clauses = splits['val']['clause_text'].tolist()
    test_clauses = splits['test']['clause_text'].tolist()
    
    val_risk_labels = trainer.risk_discovery.get_risk_labels(val_clauses)
    test_risk_labels = trainer.risk_discovery.get_risk_labels(test_clauses)
    
    val_dataset = LegalClauseDataset(
        clauses=val_clauses,
        risk_labels=val_risk_labels,
        severity_scores=trainer._generate_synthetic_scores(val_clauses, 'severity'),
        importance_scores=trainer._generate_synthetic_scores(val_clauses, 'importance'),
        tokenizer=trainer.tokenizer,
        max_length=config.max_sequence_length
    )
    
    test_dataset = LegalClauseDataset(
        clauses=test_clauses,
        risk_labels=test_risk_labels,
        severity_scores=trainer._generate_synthetic_scores(test_clauses, 'severity'),
        importance_scores=trainer._generate_synthetic_scores(test_clauses, 'importance'),
        tokenizer=trainer.tokenizer,
        max_length=config.max_sequence_length
    )
    
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    
    print(f" Data loaded: {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    print("\n" + "=" * 80)
    print("  PHASE 1: CALIBRATION")
    print("=" * 80)
    
    calibrator = CalibrationFramework(model, config.device)
    
    print("\n Pre-calibration metrics:")
    ece_before = calibrator.calculate_ece(test_loader)
    mce_before = calibrator.calculate_mce(test_loader)
    
    print("\n Calibrating model...")
    optimal_temp = calibrator.temperature_scaling(val_loader)
    
    print("\n Post-calibration metrics:")
    ece_after = calibrator.calculate_ece(test_loader)
    mce_after = calibrator.calculate_mce(test_loader)
    
    print("\n" + "=" * 80)
    print(" SAVING RESULTS")
    print("=" * 80)
    
    calibration_results = {
        'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimal_temperature': optimal_temp,
        'metrics': {
            'pre_calibration': {
                'ece': float(ece_before),
                'mce': float(mce_before)
            },
            'post_calibration': {
                'ece': float(ece_after),
                'mce': float(mce_after)
            },
            'improvement': {
                'ece': float(ece_before - ece_after),
                'mce': float(mce_before - mce_after)
            }
        }
    }
    
    results_path = os.path.join(config.checkpoint_dir, 'calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    print(f" Results saved to: {results_path}")
    
    calibrated_model_path = os.path.join(config.model_save_path, 'calibrated_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'discovered_patterns': checkpoint['discovered_patterns'],
        'temperature': optimal_temp,
        'calibration_results': calibration_results
    }, calibrated_model_path)
    
    print(f" Calibrated model saved to: {calibrated_model_path}")
    
    print("\n" + "=" * 80)
    print(" CALIBRATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n Calibration Results:")
    print(f"  Optimal Temperature: {optimal_temp:.4f}")
    print(f"\n  ECE Improvement: {ece_before:.4f} → {ece_after:.4f} (Δ {ece_before - ece_after:.4f})")
    print(f"  MCE Improvement: {mce_before:.4f} → {mce_after:.4f} (Δ {mce_before - mce_after:.4f})")
    
    if ece_after < 0.08:
        print(f"\n   Target ECE (<0.08) achieved!")
    else:
        print(f"\n    ECE slightly above target (0.08)")
    
    print(f"\n Next Steps:")
    print(f"  1. Analyze calibration quality across risk categories")
    print(f"  2. Compare with baseline methods")
    print(f"  3. Generate final implementation report")
    
    return calibrator, calibration_results

if __name__ == "__main__":
    calibrator, results = main()
