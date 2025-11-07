import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple

class FullyLearningBasedLegalBERT(nn.Module):    
    def __init__(self, config, num_discovered_risks: int = 7):
        super().__init__()
        self.config = config
        self.num_discovered_risks = num_discovered_risks
        
        try:
            self.bert = AutoModel.from_pretrained(config.bert_model_name)
            self.bert.config.hidden_dropout_prob = config.dropout_rate
            self.bert.config.attention_probs_dropout_prob = config.dropout_rate
        except:
            print(" Warning: Using mock BERT model (transformers not available)")
            self.bert = None
        
        hidden_size = 768
        
        self.risk_classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 2, num_discovered_risks)
        )
        
        self.severity_regressor = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.importance_regressor = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                output_attentions: bool = False) -> Dict[str, torch.Tensor]:
        if self.bert is not None:
            outputs = self.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            pooled_output = outputs.pooler_output
            attentions = outputs.attentions if output_attentions else None
        else:
            batch_size = input_ids.size(0)
            pooled_output = torch.randn(batch_size, 768)
            if input_ids.is_cuda:
                pooled_output = pooled_output.cuda()
            attentions = None
        
        risk_logits = self.risk_classifier(pooled_output)
        severity_score = self.severity_regressor(pooled_output).squeeze(-1) * 10
        importance_score = self.importance_regressor(pooled_output).squeeze(-1) * 10
        
        calibrated_logits = risk_logits / self.temperature
        
        result = {
            'risk_logits': risk_logits,
            'calibrated_logits': calibrated_logits,
            'severity_score': severity_score,
            'importance_score': importance_score,
            'pooled_output': pooled_output
        }
        
        if output_attentions and attentions is not None:
            result['attentions'] = attentions
        
        return result
    
    def predict_risk_pattern(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            return_attentions: bool = False) -> Dict[str, Any]:
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, output_attentions=return_attentions)
            
            risk_probs = torch.softmax(outputs['calibrated_logits'], dim=-1)
            predicted_risk = torch.argmax(risk_probs, dim=-1)
            confidence = torch.max(risk_probs, dim=-1)[0]
            
            result = {
                'predicted_risk_id': predicted_risk.cpu().numpy(),
                'risk_probabilities': risk_probs.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'severity_score': outputs['severity_score'].cpu().numpy(),
                'importance_score': outputs['importance_score'].cpu().numpy()
            }
            
            if return_attentions and 'attentions' in outputs:
                result['attentions'] = outputs['attentions']
            
            return result
    
    def analyze_attention(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                         tokenizer: Optional['LegalBertTokenizer'] = None) -> Dict[str, Any]:
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, output_attentions=True)
            
            if 'attentions' not in outputs or outputs['attentions'] is None:
                return {'error': 'Attention weights not available'}
            
            attentions = outputs['attentions']
            batch_size, seq_len = input_ids.shape
            
            all_attentions = torch.stack(attentions)
            
            cls_attention = all_attentions[:, :, :, 0, :].mean(dim=[0, 2])
            
            global_attention = all_attentions.mean(dim=[0, 2, 3])
            
            token_importance = (cls_attention + global_attention) / 2
            
            token_importance = token_importance * attention_mask
            
            k = min(10, seq_len)
            top_values, top_indices = torch.topk(token_importance, k, dim=1)
            
            result = {
                'token_importance': token_importance.cpu().numpy(),
                'top_token_indices': top_indices.cpu().numpy(),
                'top_token_scores': top_values.cpu().numpy(),
                'attention_weights': {
                    'cls_attention': cls_attention.cpu().numpy(),
                    'global_attention': global_attention.cpu().numpy()
                }
            }
            
            layer_attentions = []
            for layer_idx, layer_attn in enumerate(attentions):
                layer_cls_attn = layer_attn[:, :, 0, :].mean(dim=1)
                layer_attentions.append({
                    'layer': layer_idx,
                    'cls_attention': layer_cls_attn.cpu().numpy()
                })
            result['layer_analysis'] = layer_attentions
            
            if tokenizer is not None and tokenizer.tokenizer is not None:
                tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids[0])
                top_tokens = [tokens[idx] for idx in top_indices[0].cpu().numpy()]
                result['tokens'] = tokens
                result['top_tokens'] = top_tokens
            
            return result

class LegalBertTokenizer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            print(" Warning: Using mock tokenizer (transformers not available)")
            self.tokenizer = None
    
    def tokenize_clauses(self, clauses: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize legal clauses for model input"""
        
        if self.tokenizer is None:
            batch_size = len(clauses)
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, max_length)),
                'attention_mask': torch.ones(batch_size, max_length)
            }
        
        encoded = self.tokenizer(
            clauses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def decode_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text"""
        if self.tokenizer is None:
            return ["Mock decoded text"] * token_ids.size(0)
        
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)



class HierarchicalLegalBERT(nn.Module):
    
    def __init__(
        self,
        config,
        num_discovered_risks: int = 7,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2
    ):
        super().__init__()
        self.config = config
        self.num_discovered_risks = num_discovered_risks
        self.hidden_dim = hidden_dim
        
        try:
            self.bert = AutoModel.from_pretrained(config.bert_model_name)
            self.bert.config.hidden_dropout_prob = config.dropout_rate
            self.bert.config.attention_probs_dropout_prob = config.dropout_rate
            self.bert_hidden_size = self.bert.config.hidden_size
        except:
            print(" Warning: Using mock BERT model")
            self.bert = None
            self.bert_hidden_size = 768
        
        self.clause_to_section = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=config.dropout_rate if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.section_to_document = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=config.dropout_rate if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.clause_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        self.section_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        self.risk_classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim, num_discovered_risks)
        )
        
        self.severity_regressor = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.importance_regressor = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
    
    def encode_clause(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.bert is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.pooler_output
        else:
            batch_size = input_ids.size(0)
            return torch.randn(batch_size, self.bert_hidden_size).to(input_ids.device)
    
    def forward_single_clause(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        clause_embedding = self.encode_clause(input_ids, attention_mask)
        
        lstm_out, _ = self.clause_to_section(clause_embedding.unsqueeze(1))
        context_aware_repr = lstm_out.squeeze(1)
        
        risk_logits = self.risk_classifier(context_aware_repr)
        severity_score = self.severity_regressor(context_aware_repr).squeeze(-1) * 10
        importance_score = self.importance_regressor(context_aware_repr).squeeze(-1) * 10
        calibrated_logits = risk_logits / self.temperature
        
        return {
            'risk_logits': risk_logits,
            'calibrated_logits': calibrated_logits,
            'severity_score': severity_score,
            'importance_score': importance_score,
            'pooled_output': context_aware_repr
        }
    
    def forward_document(
        self,
        document_structure: List[List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        section_vectors = []
        all_clause_predictions = []
        attention_weights = {'clause': [], 'section': None}
        
        for section_idx, section_clauses in enumerate(document_structure):
            if not section_clauses:
                continue
            
            clause_embeddings = []
            for clause_input in section_clauses:
                input_ids = clause_input['input_ids'].unsqueeze(0).to(device)
                attention_mask = clause_input['attention_mask'].unsqueeze(0).to(device)
                clause_emb = self.encode_clause(input_ids, attention_mask)
                clause_embeddings.append(clause_emb)
            
            clause_hidden = torch.cat(clause_embeddings, dim=0)
            
            clause_lstm_out, _ = self.clause_to_section(clause_hidden.unsqueeze(0))
            
            attention_logits = self.clause_attention(clause_lstm_out)
            clause_attn = F.softmax(attention_logits, dim=1)
            section_vec = torch.sum(clause_lstm_out * clause_attn, dim=1)
            
            section_vectors.append(section_vec)
            attention_weights['clause'].append(clause_attn.squeeze(0))
            
            for i in range(len(section_clauses)):
                clause_repr = clause_lstm_out[0, i, :]
                
                risk_logits = self.risk_classifier(clause_repr)
                severity = self.severity_regressor(clause_repr).squeeze() * 10
                importance = self.importance_regressor(clause_repr).squeeze() * 10
                calibrated_logits = risk_logits / self.temperature
                
                all_clause_predictions.append({
                    'risk_logits': risk_logits,
                    'calibrated_logits': calibrated_logits,
                    'severity_score': severity,
                    'importance_score': importance,
                    'section_idx': section_idx,
                    'clause_idx': i
                })
        
        if section_vectors:
            section_hidden = torch.cat(section_vectors, dim=0)
            section_lstm_out, _ = self.section_to_document(section_hidden.unsqueeze(0))
            
            attention_logits = self.section_attention(section_lstm_out)
            section_attn = F.softmax(attention_logits, dim=1)
            document_vec = torch.sum(section_lstm_out * section_attn, dim=1)
            
            attention_weights['section'] = section_attn.squeeze(0)
        else:
            document_vec = torch.zeros(1, self.hidden_dim * 2).to(device)
        
        return {
            'document_embedding': document_vec,
            'clause_predictions': all_clause_predictions,
            'attention_weights': attention_weights
        }
    
    def predict_document(
        self,
        document_structure: List[List[Dict[str, torch.Tensor]]]
    ) -> Dict[str, Any]:
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward_document(document_structure)
        
        predictions = []
        for pred in outputs['clause_predictions']:
            risk_probs = F.softmax(pred['calibrated_logits'], dim=0).cpu().numpy()
            predicted_risk = int(risk_probs.argmax())
            
            predictions.append({
                'section_idx': pred['section_idx'],
                'clause_idx': pred['clause_idx'],
                'predicted_risk_id': predicted_risk,
                'risk_probabilities': risk_probs.tolist(),
                'confidence': float(risk_probs[predicted_risk]),
                'severity_score': pred['severity_score'].item(),
                'importance_score': pred['importance_score'].item()
            })
        
        return {
            'clauses': predictions,
            'attention_weights': {
                'clause': [attn.cpu().numpy().tolist() for attn in outputs['attention_weights']['clause']],
                'section': outputs['attention_weights']['section'].cpu().numpy().tolist() 
                          if outputs['attention_weights']['section'] is not None else None
            },
            'summary': {
                'num_sections': len(document_structure),
                'num_clauses': len(predictions),
                'avg_severity': sum(p['severity_score'] for p in predictions) / len(predictions) if predictions else 0,
                'high_risk_count': sum(1 for p in predictions if p['severity_score'] > 7)
            }
        }