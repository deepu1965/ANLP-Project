from dataclasses import dataclass
from typing import Dict, Any
import torch

@dataclass
class LegalBertConfig:
    bert_model_name: str = "bert-base-uncased"
    num_risk_categories: int = 7
    max_sequence_length: int = 512
    dropout_rate: float = 0.1
    
    hierarchical_hidden_dim: int = 512
    hierarchical_num_lstm_layers: int = 2
    
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    
    task_weights: Dict[str, float] = None
    
    use_focal_loss: bool = True
    focal_loss_gamma: float = 2.5
    minority_class_boost: float = 1.8
    
    use_lr_scheduler: bool = True
    scheduler_pct_start: float = 0.1
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path: str = "dataset/CUAD_v1/CUAD_v1.json"
    model_save_path: str = "models/legal_bert"
    checkpoint_dir: str = "checkpoints"
    
    risk_discovery_method: str = "lda"
    risk_discovery_clusters: int = 7
    tfidf_max_features: int = 15000
    tfidf_ngram_range: tuple = (1, 3)
    
    lda_doc_topic_prior: float = 0.1
    lda_topic_word_prior: float = 0.01
    lda_max_iter: int = 50
    lda_max_features: int = 8000
    lda_learning_method: str = 'batch'
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'classification': 20.0,
                'severity': 0.5,
                'importance': 0.5
            }

config = LegalBertConfig()