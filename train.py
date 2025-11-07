import torch
import os
import json
import argparse
from datetime import datetime

from config import LegalBertConfig
from trainer import LegalBertTrainer
from utils import set_seed, plot_training_history

def main():
    
    parser = argparse.ArgumentParser(description='Train Hierarchical Legal-BERT model')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    args = parser.parse_args()
    
    print("=" * 80)
    print("  HIERARCHICAL LEGAL-BERT TRAINING PIPELINE")
    print("=" * 80)
    
    config = LegalBertConfig()
    
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    set_seed(42)
    
    print(f"\n Configuration:")
    print(f"  Model type: Hierarchical BERT (context-aware)")
    print(f"  Data path: {config.data_path}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Risk discovery clusters: {config.risk_discovery_clusters}")
    print(f"  Hierarchical hidden dim: {config.hierarchical_hidden_dim}")
    print(f"  Hierarchical LSTM layers: {config.hierarchical_num_lstm_layers}")
    
    trainer = LegalBertTrainer(config)
    
    print("\n" + "=" * 80)
    print(" PHASE 1: DATA PREPARATION & RISK DISCOVERY")
    print("=" * 80)
    
    try:
        train_loader, val_loader, test_loader = trainer.prepare_data(config.data_path)
    except FileNotFoundError:
        print(f" Error: Dataset not found at {config.data_path}")
        print("Please ensure CUAD dataset is downloaded and path is correct.")
        return None, None
    except Exception as e:
        print(f" Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    print("\n Discovered Risk Patterns:")
    for pattern_name, pattern_info in trainer.risk_discovery.discovered_patterns.items():
        print(f"  • {pattern_name}")
        print(f"    Keywords: {', '.join(pattern_info['keywords'][:5])}")
    
    print("\n" + "=" * 80)
    print("  PHASE 2: MODEL TRAINING")
    print("=" * 80)
    
    try:
        history = trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f" Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    print("\n Plotting training history...")
    plot_training_history(history, save_path=os.path.join(config.checkpoint_dir, 'training_history.png'))
    
    print("\n Saving final model...")
    final_model_path = os.path.join(config.model_save_path, 'final_model.pt')
    os.makedirs(config.model_save_path, exist_ok=True)
    
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_type': 'hierarchical',
        'config': config,
        'risk_discovery_model': trainer.risk_discovery,
        'discovered_patterns': trainer.risk_discovery.discovered_patterns,
        'training_history': history
    }, final_model_path)
    
    print(f" Model saved to: {final_model_path}")
    
    summary = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'device': config.device
        },
        'final_metrics': {
            'train_loss': history['train_loss'][-1],
            'val_loss': history['val_loss'][-1],
            'train_acc': history['train_acc'][-1],
            'val_acc': history['val_acc'][-1]
        },
        'num_discovered_risks': trainer.risk_discovery.n_clusters,
        'discovered_patterns': list(trainer.risk_discovery.discovered_patterns.keys())
    }
    
    summary_path = os.path.join(config.checkpoint_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n Training summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n Final Results:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"\n Next Steps:")
    print(f"  1. Run evaluation: python evaluate.py")
    print(f"  2. Apply calibration methods")
    print(f"  3. Generate comprehensive analysis report")
    
    return trainer, history

if __name__ == "__main__":
    result = main()
    if result is not None:
        trainer, history = result
    else:
        print("\n Training failed. Please check errors above.")
        exit(1)
