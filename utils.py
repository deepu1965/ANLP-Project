import os
import json
import re
from typing import Dict, List, Any, Tuple
import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('legal_bert.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f" Created directory: {path}")

def save_json(data: Dict[str, Any], filepath: str):
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f" Saved JSON: {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f" Loaded JSON: {filepath}")
    return data

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[^\w\s.,;:()"-]', ' ', text)
    
    text = text.strip()
    
    return text

def extract_contract_metadata(filename: str) -> Dict[str, str]:
    parts = filename.replace('.txt', '').split('_')
    
    metadata = {
        'company': parts[0] if len(parts) > 0 else 'Unknown',
        'date': parts[1] if len(parts) > 1 else 'Unknown',
        'filing_type': parts[2] if len(parts) > 2 else 'Unknown',
        'exhibit': parts[3] if len(parts) > 3 else 'Unknown',
        'agreement_type': '_'.join(parts[4:]) if len(parts) > 4 else 'Unknown'
    }
    
    return metadata

def format_risk_score(score: float) -> str:
    if score < 2:
        return f"LOW ({score:.2f})"
    elif score < 5:
        return f"MEDIUM ({score:.2f})"
    elif score < 8:
        return f"HIGH ({score:.2f})"
    else:
        return f"CRITICAL ({score:.2f})"

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    import statistics
    
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'median': statistics.median(values)
    }

def set_seed(seed: int = 42):
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f" Random seed set to {seed}")
    except ImportError:
        print(f" Random seed set to {seed} (torch not available)")

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Training history plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print(" matplotlib not available. Skipping training history plot.")

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def print_progress_bar(iteration: int, total: int, prefix: str = 'Progress', 
                      suffix: str = 'Complete', length: int = 50):
    percent = (100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='')
    if iteration == total:
        print()

def validate_config(config) -> List[str]:
    errors = []
    
    required_fields = ['bert_model_name', 'data_path', 'batch_size', 'num_epochs']
    for field in required_fields:
        if not hasattr(config, field):
            errors.append(f"Missing required config field: {field}")
    
    if hasattr(config, 'data_path') and not os.path.exists(config.data_path):
        errors.append(f"Data path does not exist: {config.data_path}")
    
    if hasattr(config, 'batch_size') and config.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if hasattr(config, 'num_epochs') and config.num_epochs <= 0:
        errors.append("Number of epochs must be positive")

    if hasattr(config, 'learning_rate') and (config.learning_rate <= 0 or config.learning_rate > 1):
        errors.append("Learning rate must be between 0 and 1")
    
    return errors

def create_model_summary(model, config) -> str:
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except:
        total_params = "Unknown"
        trainable_params = "Unknown"
    
    summary = [
        " MODEL SUMMARY",
        "=" * 50,
        f"Architecture: Legal-BERT (Fully Learning-Based)",
        f"Base Model: {config.bert_model_name}",
        f"Risk Categories: {config.num_risk_categories} (discovered)",
        f"Max Sequence Length: {config.max_sequence_length}",
        f"Dropout Rate: {config.dropout_rate}",
        f"Total Parameters: {total_params}",
        f"Trainable Parameters: {trainable_params}",
        f"Device: {config.device}",
        "=" * 50
    ]
    
    return "\n".join(summary)

def check_dependencies() -> Dict[str, bool]:
    dependencies = {
        'torch': False,
        'transformers': False,
        'sklearn': False,
        'numpy': False,
        'pandas': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

def print_dependency_status():
    deps = check_dependencies()
    
    print(" DEPENDENCY STATUS")
    print("-" * 30)
    
    for dep, available in deps.items():
        status = " Available" if available else " Missing"
        print(f"{dep:12} : {status}")
    
    missing = [dep for dep, available in deps.items() if not available]
    
    if missing:
        print(f"\n  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch transformers scikit-learn numpy pandas")
        print("For demo mode, dependencies are not required.")
    else:
        print("\n All dependencies available!")

def get_sample_contract_text() -> str:
    return """
    SERVICES AGREEMENT
    
    This Services Agreement ("Agreement") is entered into as of the Effective Date
    by and between Company A ("Provider") and Company B ("Client").
    
    1. SERVICES
    Provider shall provide the services described in Exhibit A ("Services") to Client
    in accordance with the terms and conditions set forth herein.
    
    2. PAYMENT TERMS
    Client shall pay Provider the fees specified in Exhibit B within thirty (30) days
    of receipt of each invoice. Late payments shall incur a penalty of 1.5% per month.
    
    3. INDEMNIFICATION
    Each party shall indemnify and hold harmless the other party from and against any
    third-party claims arising out of such party's breach of this Agreement.
    
    4. LIMITATION OF LIABILITY
    In no event shall either party's liability exceed the total amount paid under this
    Agreement in the twelve (12) months preceding the claim.
    
    5. TERMINATION
    Either party may terminate this Agreement upon thirty (30) days written notice
    to the other party. Upon termination, all confidential information shall be returned.
    
    6. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws
    of the State of Delaware.
    """


def split_into_clauses(text: str, method: str = 'sentence') -> List[str]:
    if not text or not isinstance(text, str):
        return []
    
    if method == 'sentence':
        import re

        clauses = re.split(r'(?<=[.;])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z])', text)
        
        clauses = [c.strip() for c in clauses if c.strip()]

        clauses = [c for c in clauses if len(c) >= 10]
        
        return clauses
    
    elif method == 'legal':
        import re
        
        clauses = []
        
        sections = re.split(r'\n\s*(\d+\.?\s+[A-Z][A-Z\s]+)\n', text)
        
        for section in sections:
            if not section.strip():
                continue
            sentences = re.split(r'(?<=[.;])\s+(?=[A-Z(])', section)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) >= 10:
                    clauses.append(sent)
        
        return clauses
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sentence' or 'legal'")


def analyze_full_document(
    text: str, 
    model, 
    return_details: bool = True,
    use_context: bool = True,
    context_window: int = 1
) -> Dict[str, Any]:

    clauses = split_into_clauses(text, method='legal')
    
    if not clauses:
        return {
            'error': 'No clauses found in document',
            'n_clauses': 0
        }
    
    
    clause_predictions = []
    
    if use_context:
        print(f" Analyzing document with {len(clauses)} clauses (context-aware)...")
        print(f"   Context window: ±{context_window} clauses")
    else:
        print(f" Analyzing document with {len(clauses)} clauses...")
    
    for i, clause in enumerate(clauses):
        try:
            
            if use_context:

                start_idx = max(0, i - context_window)
                end_idx = min(len(clauses), i + context_window + 1)

                context_clauses = clauses[start_idx:end_idx]

                clause_with_context = " ".join(context_clauses)
                
                
                input_text = clause_with_context
            else:
                input_text = clause
            
            pred = model.predict(input_text)
            
            clause_predictions.append({
                'clause_id': i,
                'clause_text': clause,  
                'analyzed_with_context': use_context,
                'risk_type': pred.get('risk_type'),
                'risk_name': pred.get('risk_name'),
                'confidence': pred.get('confidence'),
                'severity': pred.get('severity'),
                'importance': pred.get('importance')
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(clauses)} clauses...")
                
        except Exception as e:
            print(f"  Error analyzing clause {i}: {e}")
            continue
    
    if not clause_predictions:
        return {
            'error': 'Failed to analyze any clauses',
            'n_clauses': len(clauses)
        }
    
    severities = [p['severity'] for p in clause_predictions if p.get('severity')]
    importances = [p['importance'] for p in clause_predictions if p.get('importance')]
    
    high_risk_clauses = [
        p for p in clause_predictions 
        if p.get('severity', 0) > 7.0
    ]
    
    from collections import Counter
    risk_counts = Counter([p['risk_name'] for p in clause_predictions if p.get('risk_name')])
    total = len(clause_predictions)
    risk_distribution = {
        risk: count / total 
        for risk, count in risk_counts.items()
    }
    
    dominant_risk = risk_counts.most_common(1)[0] if risk_counts else ('UNKNOWN', 0)

    result = {
        'document_summary': {
            'total_clauses': len(clauses),
            'analyzed_clauses': len(clause_predictions),
            'overall_severity': sum(severities) / len(severities) if severities else 0,
            'max_severity': max(severities) if severities else 0,
            'overall_importance': sum(importances) / len(importances) if importances else 0,
            'high_risk_clause_count': len(high_risk_clauses),
            'dominant_risk_type': dominant_risk[0],
            'dominant_risk_percentage': (dominant_risk[1] / total * 100) if total > 0 else 0
        },
        'risk_distribution': risk_distribution,
        'high_risk_clauses': high_risk_clauses[:10] if high_risk_clauses else []  
    }

    if return_details:
        result['all_clauses'] = clause_predictions
    
    print(f" Analysis complete!")
    print(f"   Overall Severity: {result['document_summary']['overall_severity']:.2f}")
    print(f"   High-Risk Clauses: {len(high_risk_clauses)}")
    print(f"   Dominant Risk: {dominant_risk[0]} ({dominant_risk[1]} clauses)")
    
    return result


def analyze_with_section_context(text: str, model, return_details: bool = True) -> Dict[str, Any]:

    import re
    
    print(" Analyzing document with section-aware context...")
    section_pattern = r'\n\s*(\d+\.?\d*\s+[A-Z][A-Z\s]+)\n'
    
    parts = re.split(section_pattern, text)
    
    sections = []
    current_section = {'title': 'Preamble', 'text': parts[0], 'clauses': []}

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            if current_section['text'].strip():
                section_clauses = split_into_clauses(current_section['text'], method='sentence')
                current_section['clauses'] = section_clauses
                sections.append(current_section)

            current_section = {
                'title': parts[i].strip(),
                'text': parts[i + 1],
                'clauses': []
            }

    if current_section['text'].strip():
        section_clauses = split_into_clauses(current_section['text'], method='sentence')
        current_section['clauses'] = section_clauses
        sections.append(current_section)
    
    print(f"   Identified {len(sections)} sections")
    all_predictions = []
    section_summaries = []
    
    for sect_idx, section in enumerate(sections):
        section_title = section['title']
        section_text = section['text']
        clauses = section['clauses']
        
        print(f"   Analyzing section: {section_title} ({len(clauses)} clauses)")
        
        section_predictions = []
        
        for clause_idx, clause in enumerate(clauses):
            try:
               
                context_input = f"{section_title}. {section_text}"
                if len(context_input) > 1000:  
                    window_start = max(0, clause_idx - 2)
                    window_end = min(len(clauses), clause_idx + 3)
                    nearby = " ".join(clauses[window_start:window_end])
                    context_input = f"{section_title}. {nearby}"

                pred = model.predict(context_input)
                
                prediction = {
                    'clause_id': len(all_predictions),
                    'section': section_title,
                    'clause_text': clause,
                    'risk_type': pred.get('risk_type'),
                    'risk_name': pred.get('risk_name'),
                    'confidence': pred.get('confidence'),
                    'severity': pred.get('severity'),
                    'importance': pred.get('importance'),
                    'analyzed_with_section_context': True
                }
                
                section_predictions.append(prediction)
                all_predictions.append(prediction)
                
            except Exception as e:
                print(f"  Error in {section_title}, clause {clause_idx}: {e}")
                continue

        if section_predictions:
            severities = [p['severity'] for p in section_predictions if p.get('severity')]
            avg_severity = sum(severities) / len(severities) if severities else 0
            
            section_summaries.append({
                'title': section_title,
                'clause_count': len(clauses),
                'avg_severity': avg_severity,
                'max_severity': max(severities) if severities else 0,
                'high_risk_count': sum(1 for s in severities if s > 7)
            })
    if not all_predictions:
        return {'error': 'No predictions generated'}
    
    from collections import Counter
    
    severities = [p['severity'] for p in all_predictions if p.get('severity')]
    risk_counts = Counter([p['risk_name'] for p in all_predictions if p.get('risk_name')])
    total = len(all_predictions)
    
    result = {
        'document_summary': {
            'total_sections': len(sections),
            'total_clauses': len(all_predictions),
            'overall_severity': sum(severities) / len(severities) if severities else 0,
            'max_severity': max(severities) if severities else 0,
            'high_risk_clause_count': sum(1 for s in severities if s > 7)
        },
        'sections': section_summaries,
        'risk_distribution': {risk: count/total for risk, count in risk_counts.items()},
        'all_clauses': all_predictions if return_details else []
    }
    
    print(f" Analysis complete!")
    print(f"   {len(sections)} sections analyzed")
    print(f"   Overall severity: {result['document_summary']['overall_severity']:.2f}")
    
    return result


def print_document_analysis(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print(" DOCUMENT RISK ANALYSIS REPORT")
    print("=" * 80)
    
    summary = results.get('document_summary', {})
    
    print(f"\n Document Overview:")
    print(f"   Total Clauses: {summary.get('total_clauses', 0)}")
    print(f"   Analyzed: {summary.get('analyzed_clauses', 0)}")
    
    print(f"\n  Risk Assessment:")
    severity = summary.get('overall_severity', 0)
    print(f"   Overall Severity: {severity:.2f}/10 - {format_risk_score(severity)}")
    print(f"   Maximum Severity: {summary.get('max_severity', 0):.2f}/10")
    print(f"   Overall Importance: {summary.get('overall_importance', 0):.2f}/10")
    
    print(f"\n High-Risk Clauses:")
    print(f"   Count: {summary.get('high_risk_clause_count', 0)}")
    
    print(f"\n Risk Distribution:")
    for risk_type, percentage in results.get('risk_distribution', {}).items():
        print(f"   {risk_type}: {percentage*100:.1f}%")
    
    print(f"\n Dominant Risk:")
    print(f"   {summary.get('dominant_risk_type', 'N/A')} "
          f"({summary.get('dominant_risk_percentage', 0):.1f}% of clauses)")

    high_risk = results.get('high_risk_clauses', [])
    if high_risk:
        print(f"\n Top High-Risk Clauses:")
        for i, clause in enumerate(high_risk[:5], 1):
            print(f"\n   {i}. {clause['risk_name']} (Severity: {clause['severity']:.1f})")
            text = clause['clause_text'][:100] + "..." if len(clause['clause_text']) > 100 else clause['clause_text']
            print(f"      \"{text}\"")
    
    print("\n" + "=" * 80)


def parse_document_hierarchically(text: str) -> List[List[str]]:
    section_pattern = r'\n\s*(\d+\.?\d*\s+[A-Z][A-Z\s]+)\n'
    sections = re.split(section_pattern, text)
    
    document_structure = []
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            section_title = sections[i].strip()
            section_text = sections[i + 1].strip()
            clauses = split_into_clauses(section_text, method='sentence')
            
            if clauses:
                document_structure.append(clauses)
    if not document_structure:
        clauses = split_into_clauses(text, method='sentence')
        if clauses:
            document_structure.append(clauses)
    
    return document_structure


def prepare_hierarchical_input(clauses: List[str], tokenizer) -> List[Dict[str, Any]]:
    clause_inputs = []
    
    for clause in clauses:
        encoded = tokenizer.tokenize_clauses([clause], max_length=128)
        clause_inputs.append({
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        })
    
    return clause_inputs