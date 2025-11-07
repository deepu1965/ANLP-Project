import torch
import os
import json
from datetime import datetime

from config import LegalBertConfig
from trainer import LegalBertTrainer, collate_batch
from evaluator import LegalBertEvaluator
from data_loader import CUADDataLoader
from risk_discovery import UnsupervisedRiskDiscovery

def main():
    print("=" * 80)
    print(" LEGAL-BERT EVALUATION PIPELINE")
    print("=" * 80)
    
    config = LegalBertConfig()
    
    print("\n Loading trained model...")
    model_path = os.path.join(config.model_save_path, 'final_model.pt')
    
    if not os.path.exists(model_path):
        print(f" Error: Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    
    trainer = LegalBertTrainer(config)
    
    if 'risk_discovery_model' in checkpoint:
        trainer.risk_discovery = checkpoint['risk_discovery_model']
    else:
        trainer.risk_discovery.discovered_patterns = checkpoint['discovered_patterns']
        trainer.risk_discovery.n_clusters = len(checkpoint['discovered_patterns'])
    
    from model import HierarchicalLegalBERT
    
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
    trainer.model = HierarchicalLegalBERT(
        config=config,
        num_discovered_risks=trainer.risk_discovery.n_clusters,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers
    ).to(config.device)
    
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    print(" Model loaded successfully!")
    
    print("\n Loading test data...")
    data_loader = CUADDataLoader(config.data_path)
    df_clauses, contracts = data_loader.load_data()
    splits = data_loader.create_splits()
    
    test_clauses = splits['test']['clause_text'].tolist()
    risk_labels = trainer.risk_discovery.get_risk_labels(test_clauses)
    severity_scores = trainer._generate_synthetic_scores(test_clauses, 'severity')
    importance_scores = trainer._generate_synthetic_scores(test_clauses, 'importance')
    
    from trainer import LegalClauseDataset
    from torch.utils.data import DataLoader
    
    test_dataset = LegalClauseDataset(
        clauses=test_clauses,
        risk_labels=risk_labels,
        severity_scores=severity_scores,
        importance_scores=importance_scores,
        tokenizer=trainer.tokenizer,
        max_length=config.max_sequence_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch
    )
    
    print(f" Test data prepared: {len(test_dataset)} samples")
    
    print("\n" + "=" * 80)
    print(" PHASE 1: MODEL EVALUATION")
    print("=" * 80)
    
    evaluator = LegalBertEvaluator(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        risk_discovery=trainer.risk_discovery
    )
    
    results = evaluator.evaluate_model(test_loader, save_results=True)
    
    print("\n" + "=" * 80)
    print(" EVALUATION REPORT")
    print("=" * 80)
    
    report = evaluator.generate_report()
    print(report)
    
    results_path = os.path.join(config.checkpoint_dir, 'evaluation_results.json')
    
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n Detailed results saved to: {results_path}")
    
    print("\n Generating visualizations...")
    evaluator.plot_confusion_matrix(save_path=os.path.join(config.checkpoint_dir, 'confusion_matrix.png'))
    evaluator.plot_risk_distribution(save_path=os.path.join(config.checkpoint_dir, 'risk_distribution.png'))
    
    print("\n" + "=" * 80)
    print(" EVALUATION COMPLETE!")
    print("=" * 80)
    
    clf_metrics = results['classification_metrics']
    print(f"\n Key Metrics:")
    print(f"  Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {clf_metrics['f1_score']:.4f}")
    print(f"  Precision: {clf_metrics['precision']:.4f}")
    print(f"  Recall: {clf_metrics['recall']:.4f}")
    
    reg_metrics = results['regression_metrics']
    print(f"\n Regression Performance:")
    print(f"  Severity R²: {reg_metrics['severity']['r2_score']:.4f}")
    print(f"  Importance R²: {reg_metrics['importance']['r2_score']:.4f}")
    
    print(f"\n Next Steps:")
    print(f"  1. Apply calibration methods: python calibrate.py")
    print(f"  2. Analyze error cases")
    print(f"  3. Compare with baseline methods")
    
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = main()
