import torch
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print(" Warning: matplotlib/seaborn not available. Visualizations will be skipped.")

try:
    from hierarchical_risk import HierarchicalRiskAggregator, RiskDependencyAnalyzer
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    print(" Warning: hierarchical_risk module not available.")

class LegalBertEvaluator:
    def __init__(self, model, tokenizer, risk_discovery):
        self.model = model
        self.tokenizer = tokenizer
        self.risk_discovery = risk_discovery
        self.evaluation_results = {}
    
    def evaluate_model(self, test_loader, save_results: bool = True) -> Dict[str, Any]:
        print(" Starting comprehensive evaluation...")
        all_predictions = []
        all_true_labels = []
        all_severity_preds = []
        all_severity_true = []
        all_importance_preds = []
        all_importance_true = []
        all_confidences = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                device = next(self.model.parameters()).device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.model.forward_single_clause(input_ids, attention_mask)
                
                risk_probs = torch.softmax(outputs['calibrated_logits'], dim=-1)
                predicted_risk_ids = torch.argmax(risk_probs, dim=-1)
                confidences = torch.max(risk_probs, dim=-1)[0]
                
                all_predictions.extend(predicted_risk_ids.cpu().numpy())
                all_true_labels.extend(batch['risk_label'].numpy())
                all_severity_preds.extend(outputs['severity_score'].cpu().numpy())
                all_severity_true.extend(batch['severity_score'].numpy())
                all_importance_preds.extend(outputs['importance_score'].cpu().numpy())
                all_importance_true.extend(batch['importance_score'].numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        results = {
            'classification_metrics': self._calculate_classification_metrics(
                all_true_labels, all_predictions, all_confidences
            ),
            'regression_metrics': self._calculate_regression_metrics(
                all_severity_true, all_severity_preds,
                all_importance_true, all_importance_preds
            ),
            'risk_pattern_analysis': self._analyze_risk_patterns(
                all_true_labels, all_predictions
            )
        }
        
        self.evaluation_results = results
        
        if save_results:
            self.save_evaluation_results(results)
        
        print(" Evaluation complete!")
        return results
    
    def _calculate_classification_metrics(self, true_labels: List[int], 
                                        predictions: List[int], 
                                        confidences: List[float]) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        cm = confusion_matrix(true_labels, predictions)
        
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std
        }
    
    def _calculate_regression_metrics(self, severity_true: List[float], severity_pred: List[float],
                                    importance_true: List[float], importance_pred: List[float]) -> Dict[str, Any]:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        severity_mse = mean_squared_error(severity_true, severity_pred)
        severity_mae = mean_absolute_error(severity_true, severity_pred)
        severity_r2 = r2_score(severity_true, severity_pred)
        
        importance_mse = mean_squared_error(importance_true, importance_pred)
        importance_mae = mean_absolute_error(importance_true, importance_pred)
        importance_r2 = r2_score(importance_true, importance_pred)
        
        return {
            'severity': {
                'mse': severity_mse,
                'mae': severity_mae,
                'r2_score': severity_r2
            },
            'importance': {
                'mse': importance_mse,
                'mae': importance_mae,
                'r2_score': importance_r2
            }
        }
    
    def _analyze_risk_patterns(self, true_labels: List[int], predictions: List[int]) -> Dict[str, Any]:
        discovered_patterns = self.risk_discovery.discovered_patterns
        pattern_names = list(discovered_patterns.keys())
        
        true_distribution = defaultdict(int)
        pred_distribution = defaultdict(int)
        
        for label in true_labels:
            true_distribution[pattern_names[label]] += 1
        
        for pred in predictions:
            pred_distribution[pattern_names[pred]] += 1
        
        pattern_performance = {}
        for i, pattern_name in enumerate(pattern_names):
            pattern_true = [1 if label == i else 0 for label in true_labels]
            pattern_pred = [1 if pred == i else 0 for pred in predictions]
            
            if sum(pattern_true) > 0:
                precision = sum([1 for t, p in zip(pattern_true, pattern_pred) if t == 1 and p == 1]) / max(sum(pattern_pred), 1)
                recall = sum([1 for t, p in zip(pattern_true, pattern_pred) if t == 1 and p == 1]) / sum(pattern_true)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                pattern_performance[pattern_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': sum(pattern_true)
                }
        
        return {
            'true_distribution': dict(true_distribution),
            'predicted_distribution': dict(pred_distribution),
            'pattern_performance': pattern_performance,
            'discovered_patterns_info': discovered_patterns
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            raise ValueError("Must run evaluation first")
        
        results = self.evaluation_results
        
        report = []
        report.append("=" * 80)
        report.append("  LEGAL-BERT EVALUATION REPORT")
        report.append("=" * 80)
        
        report.append("\n RISK CLASSIFICATION PERFORMANCE")
        report.append("-" * 50)
        clf_metrics = results['classification_metrics']
        report.append(f"Accuracy: {clf_metrics['accuracy']:.4f}")
        report.append(f"Precision: {clf_metrics['precision']:.4f}")
        report.append(f"Recall: {clf_metrics['recall']:.4f}")
        report.append(f"F1-Score: {clf_metrics['f1_score']:.4f}")
        report.append(f"Average Confidence: {clf_metrics['avg_confidence']:.4f}")
        
        report.append("\n REGRESSION PERFORMANCE")
        report.append("-" * 50)
        reg_metrics = results['regression_metrics']
        
        report.append("Severity Prediction:")
        report.append(f"  MSE: {reg_metrics['severity']['mse']:.4f}")
        report.append(f"  MAE: {reg_metrics['severity']['mae']:.4f}")
        report.append(f"  R²: {reg_metrics['severity']['r2_score']:.4f}")
        
        report.append("Importance Prediction:")
        report.append(f"  MSE: {reg_metrics['importance']['mse']:.4f}")
        report.append(f"  MAE: {reg_metrics['importance']['mae']:.4f}")
        report.append(f"  R²: {reg_metrics['importance']['r2_score']:.4f}")
        
        report.append("\n DISCOVERED RISK PATTERNS")
        report.append("-" * 50)
        pattern_analysis = results['risk_pattern_analysis']
        
        report.append("Pattern Distribution (True vs Predicted):")
        for pattern, count in pattern_analysis['true_distribution'].items():
            pred_count = pattern_analysis['predicted_distribution'].get(pattern, 0)
            report.append(f"  {pattern}: {count} → {pred_count}")
        
        report.append("\nPattern-Specific Performance:")
        for pattern, metrics in pattern_analysis['pattern_performance'].items():
            report.append(f"  {pattern}:")
            report.append(f"    Precision: {metrics['precision']:.4f}")
            report.append(f"    Recall: {metrics['recall']:.4f}")
            report.append(f"    F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"    Support: {metrics['support']}")
        
        report.append("\n DISCOVERED PATTERN DETAILS")
        report.append("-" * 50)
        for pattern_name, details in pattern_analysis['discovered_patterns_info'].items():
            report.append(f"\n{pattern_name}:")
            if 'clause_count' in details:
                report.append(f"  Clauses: {details['clause_count']}")
            
            if 'avg_risk_intensity' in details:
                report.append(f"  Risk Intensity: {details['avg_risk_intensity']:.3f}")
            
            if 'avg_legal_complexity' in details:
                report.append(f"  Legal Complexity: {details['avg_legal_complexity']:.3f}")
            
            if 'key_terms' in details:
                report.append(f"  Key Terms: {', '.join(details['key_terms'][:5])}")
            elif 'top_words' in details:
                report.append(f"  Top Words: {', '.join(details['top_words'][:5])}")
            
            if 'topic_distribution' in details:
                report.append(f"  Topic Distribution: {details['topic_distribution']:.3f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def plot_confusion_matrix(self, save_path: str = None):
        if not VISUALIZATION_AVAILABLE:
            print(" Visualization libraries not available. Skipping plot.")
            return
        
        if not self.evaluation_results:
            raise ValueError("Must run evaluation first")
        
        cm = np.array(self.evaluation_results['classification_metrics']['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Risk Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Confusion matrix saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_risk_distribution(self, save_path: str = None):
        if not VISUALIZATION_AVAILABLE:
            print(" Visualization libraries not available. Skipping plot.")
            return
        
        if not self.evaluation_results:
            raise ValueError("Must run evaluation first")
        
        pattern_analysis = self.evaluation_results['risk_pattern_analysis']
        patterns = list(pattern_analysis['true_distribution'].keys())
        true_counts = [pattern_analysis['true_distribution'][p] for p in patterns]
        pred_counts = [pattern_analysis['predicted_distribution'].get(p, 0) for p in patterns]
        
        x = np.arange(len(patterns))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        
        ax.set_xlabel('Risk Patterns')
        ax.set_ylabel('Count')
        ax.set_title('Risk Pattern Distribution - True vs Predicted')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Risk distribution plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        json_results = self._convert_for_json(results)
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        report = self.generate_report()
        with open('evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print(" Evaluation results saved:")
        print("  - evaluation_results.json")
        print("  - evaluation_report.txt")
    
    def _convert_for_json(self, obj):
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def analyze_attention_patterns(self, test_clauses: List[str], 
                                   max_samples: int = 10) -> Dict[str, Any]:
        print(f" Analyzing attention patterns for {min(len(test_clauses), max_samples)} samples...")
        
        self.model.eval()
        attention_results = []
        
        with torch.no_grad():
            for idx, clause in enumerate(test_clauses[:max_samples]):
                tokens = self.tokenizer.tokenize_clauses([clause])
                input_ids = tokens['input_ids'].to(self.model.config.device)
                attention_mask = tokens['attention_mask'].to(self.model.config.device)
                
                analysis = self.model.analyze_attention(input_ids, attention_mask, self.tokenizer)
                
                prediction = self.model.predict_risk_pattern(input_ids, attention_mask)
                
                result = {
                    'clause_index': idx,
                    'clause_preview': clause[:100] + '...' if len(clause) > 100 else clause,
                    'predicted_risk': int(prediction['predicted_risk_id'][0]),
                    'severity': float(prediction['severity_score'][0]),
                    'importance': float(prediction['importance_score'][0]),
                    'top_tokens': analysis.get('top_tokens', []),
                    'top_token_scores': analysis.get('top_token_scores', np.array([])).tolist()
                }
                
                attention_results.append(result)
        
        print(f" Attention analysis complete for {len(attention_results)} clauses")
        
        return {
            'num_analyzed': len(attention_results),
            'clause_analyses': attention_results
        }
    
    def evaluate_hierarchical_risk(self, test_loader, 
                                   contract_ids: List[int]) -> Dict[str, Any]:
        if not HIERARCHICAL_AVAILABLE:
            print(" Hierarchical risk analysis not available")
            return {'error': 'hierarchical_risk module not found'}
        
        print(" Performing hierarchical risk evaluation (clause → contract level)...")
        
        contract_predictions = defaultdict(list)
        
        self.model.eval()
        clause_idx = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.model.config.device)
                attention_mask = batch['attention_mask'].to(self.model.config.device)
                
                predictions = self.model.predict_risk_pattern(input_ids, attention_mask)
                
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    contract_id = contract_ids[clause_idx]
                    
                    clause_pred = {
                        'predicted_risk_id': int(predictions['predicted_risk_id'][i]),
                        'confidence': float(predictions['confidence'][i]),
                        'severity_score': float(predictions['severity_score'][i]),
                        'importance_score': float(predictions['importance_score'][i])
                    }
                    
                    contract_predictions[contract_id].append(clause_pred)
                    clause_idx += 1
        
        aggregator = HierarchicalRiskAggregator()
        contract_results = {}
        
        for contract_id, clause_preds in contract_predictions.items():
            contract_risk = aggregator.aggregate_contract_risk(
                clause_preds, 
                method='weighted_mean'
            )
            contract_results[contract_id] = contract_risk
        
        print(f" Analyzed {len(contract_results)} contracts")
        
        contract_severities = [r['contract_severity'] for r in contract_results.values()]
        contract_importances = [r['contract_importance'] for r in contract_results.values()]
        
        summary = {
            'num_contracts': len(contract_results),
            'contract_results': contract_results,
            'summary_statistics': {
                'avg_contract_severity': float(np.mean(contract_severities)),
                'std_contract_severity': float(np.std(contract_severities)),
                'max_contract_severity': float(np.max(contract_severities)),
                'min_contract_severity': float(np.min(contract_severities)),
                'avg_contract_importance': float(np.mean(contract_importances)),
                'high_risk_contracts': sum(1 for s in contract_severities if s >= 7.0)
            }
        }
        
        return summary
    
    def analyze_risk_dependencies(self, test_loader, 
                                  contract_ids: List[int],
                                  num_risk_types: int = 7) -> Dict[str, Any]:
        if not HIERARCHICAL_AVAILABLE:
            print(" Risk dependency analysis not available")
            return {'error': 'hierarchical_risk module not found'}
        
        print(" Analyzing risk dependencies and interactions...")
        
        contract_predictions = defaultdict(list)
        
        self.model.eval()
        clause_idx = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.model.config.device)
                attention_mask = batch['attention_mask'].to(self.model.config.device)
                
                predictions = self.model.predict_risk_pattern(input_ids, attention_mask)
                
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    contract_id = contract_ids[clause_idx]
                    
                    clause_pred = {
                        'predicted_risk_id': int(predictions['predicted_risk_id'][i]),
                        'confidence': float(predictions['confidence'][i]),
                        'severity_score': float(predictions['severity_score'][i]),
                        'importance_score': float(predictions['importance_score'][i])
                    }
                    
                    contract_predictions[contract_id].append(clause_pred)
                    clause_idx += 1
        
        dependency_analyzer = RiskDependencyAnalyzer()
        
        contract_pred_lists = list(contract_predictions.values())
        correlation_matrix = dependency_analyzer.compute_risk_correlation(
            contract_pred_lists, 
            num_risk_types
        )
        
        all_clause_preds = [pred for preds in contract_pred_lists for pred in preds]
        amplification = dependency_analyzer.analyze_risk_amplification(all_clause_preds)
        
        all_chains = []
        for clause_preds in contract_pred_lists:
            chains = dependency_analyzer.find_risk_chains(clause_preds, window_size=3)
            all_chains.extend(chains)
        
        from collections import Counter
        chain_counts = Counter([tuple(chain) for chain in all_chains])
        most_common_chains = chain_counts.most_common(10)
        
        print(f" Risk dependency analysis complete")
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'risk_amplification': amplification,
            'common_risk_chains': [
                {'chain': list(chain), 'count': count} 
                for chain, count in most_common_chains
            ],
            'total_chains_found': len(all_chains)
        }

try:
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    print(" Warning: Some evaluation dependencies not available. Using mock implementations.")

    class MockTensor:
        def __init__(self, data):
            self.data = data
        def numpy(self):
            return self.data
        def to(self, device):
            return self
    
    class MockModule:
        def eval(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    torch = type('torch', (), {
        'no_grad': lambda: type('context', (), {'__enter__': lambda self: None, '__exit__': lambda *args: None})()
    })()

    def accuracy_score(y_true, y_pred):
        return sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)
    
    def precision_recall_fscore_support(y_true, y_pred, average=None):
        return 0.5, 0.5, 0.5, None
    
    def confusion_matrix(y_true, y_pred):
        return [[1, 0], [0, 1]]
    
    def mean_squared_error(y_true, y_pred):
        return sum([(t - p) ** 2 for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def mean_absolute_error(y_true, y_pred):
        return sum([abs(t - p) for t, p in zip(y_true, y_pred)]) / len(y_true)
    
    def r2_score(y_true, y_pred):
        return 0.5