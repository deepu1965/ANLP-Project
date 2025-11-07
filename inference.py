import torch
import json
from typing import List, Dict, Any
import argparse

from model import HierarchicalLegalBERT, LegalBertTokenizer
from config import LegalBertConfig


def load_trained_model(checkpoint_path: str, config: LegalBertConfig) -> HierarchicalLegalBERT:
    print(f" Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    
    num_risks = len(checkpoint.get('discovered_patterns', {}))
    print(f"   Model has {num_risks} discovered risk patterns")
    
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        hidden_dim = saved_config.hierarchical_hidden_dim
        num_lstm_layers = saved_config.hierarchical_num_lstm_layers
        print(f"   Using saved architecture: hidden_dim={hidden_dim}, lstm_layers={num_lstm_layers}")
    else:
        hidden_dim = config.hierarchical_hidden_dim
        num_lstm_layers = config.hierarchical_num_lstm_layers
        print(f"     Warning: No config in checkpoint, using current config")
    
    model = HierarchicalLegalBERT(
        config=config,
        num_discovered_risks=num_risks,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    print(f"    Model loaded successfully")
    
    return model, checkpoint.get('discovered_patterns', {})


def predict_single_clause(
    model: HierarchicalLegalBERT,
    tokenizer: LegalBertTokenizer,
    clause: str,
    config: LegalBertConfig
) -> Dict[str, Any]:
    
    encoded = tokenizer.tokenize_clauses([clause], config.max_sequence_length)
    input_ids = encoded['input_ids'].to(config.device)
    attention_mask = encoded['attention_mask'].to(config.device)
    
    with torch.no_grad():
        outputs = model.forward_single_clause(input_ids, attention_mask)
        
        risk_probs = torch.softmax(outputs['calibrated_logits'], dim=-1)
        predicted_risk = torch.argmax(risk_probs, dim=-1)
        confidence = torch.max(risk_probs, dim=-1)[0]
        
        return {
            'clause': clause,
            'predicted_risk_id': predicted_risk.cpu().item(),
            'confidence': confidence.cpu().item(),
            'risk_probabilities': risk_probs.cpu().numpy().tolist(),
            'severity_score': outputs['severity_score'].cpu().item(),
            'importance_score': outputs['importance_score'].cpu().item()
        }


def predict_document(
    model: HierarchicalLegalBERT,
    tokenizer: LegalBertTokenizer,
    document: List[List[str]],
    config: LegalBertConfig
) -> Dict[str, Any]:
    """
    Predict risks for a full document with context
    
    Args:
        document: List of sections, each containing list of clauses
            Example: [
                ['clause1', 'clause2'],  # Section 1
                ['clause3', 'clause4'],  # Section 2
            ]
    """
    
    print(f" Analyzing document with {len(document)} sections...")
    
    doc_structure = []
    clause_texts = []
    
    for section_idx, section in enumerate(document):
        section_tokens = []
        for clause_idx, clause in enumerate(section):
            encoded = tokenizer.tokenize_clauses([clause], config.max_sequence_length)
            section_tokens.append({
                'input_ids': encoded['input_ids'][0],
                'attention_mask': encoded['attention_mask'][0]
            })
            clause_texts.append({
                'section': section_idx,
                'clause': clause_idx,
                'text': clause
            })
        doc_structure.append(section_tokens)
    
    results = model.predict_document(doc_structure)
    
    for i, pred in enumerate(results['clauses']):
        pred['text'] = clause_texts[i]['text']
    
    return results


def format_prediction_output(
    prediction: Dict[str, Any],
    risk_patterns: Dict[str, Any]
) -> str:
    
    risk_id = prediction['predicted_risk_id']
    pattern_names = list(risk_patterns.keys())
    
    if risk_id < len(pattern_names):
        risk_name = str(pattern_names[risk_id])
        risk_info = risk_patterns[pattern_names[risk_id]]
        
        if isinstance(risk_info, dict):
            keywords = ', '.join(risk_info.get('keywords', risk_info.get('top_words', []))[:5])
        else:
            keywords = "N/A"
    else:
        risk_name = f"Risk Pattern {risk_id}"
        keywords = "N/A"
    
    output = f"""
{'='*70}
 CLAUSE ANALYSIS
{'='*70}

 Clause:
   {prediction.get('text', prediction.get('clause', 'N/A'))}

 Risk Classification:
   Pattern: {risk_name}
   Confidence: {prediction['confidence']:.1%}
   Keywords: {keywords}

 Risk Scores:
   Severity:   {prediction['severity_score']:.2f}/10
   Importance: {prediction['importance_score']:.2f}/10

 Probability Distribution:
"""
    
    probs = prediction['risk_probabilities']
    
    if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], list):
        probs = probs[0]
    
    top_3_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    
    for idx in top_3_indices:
        if idx < len(pattern_names):
            pattern_str = str(pattern_names[idx])
            if len(pattern_str) > 40:
                pattern_str = pattern_str[:37] + "..."
            output += f"   {pattern_str:40s} {probs[idx]:.1%}\n"
        else:
            output += f"   Risk Pattern {idx:2d}                          {probs[idx]:.1%}\n"
    
    return output


def main():    
    parser = argparse.ArgumentParser(description='Legal-BERT Risk Analysis Inference')
    parser.add_argument('--checkpoint', type=str, default='models/legal_bert/final_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--clause', type=str, help='Single clause to analyze')
    parser.add_argument('--document', type=str, help='Path to JSON file with document structure')
    parser.add_argument('--output', type=str, help='Path to save results (JSON)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  LEGAL-BERT RISK ANALYSIS INFERENCE")
    print("=" * 70)
    
    config = LegalBertConfig()
    print(f"\n Configuration:")
    print(f"   Device: {config.device}")
    print(f"   Max sequence length: {config.max_sequence_length}")
    
    model, risk_patterns = load_trained_model(args.checkpoint, config)
    tokenizer = LegalBertTokenizer(config.bert_model_name)
    
    print(f"\n Discovered Risk Patterns ({len(risk_patterns)}):")
    pattern_names = list(risk_patterns.keys())
    for name in pattern_names[:5]:
        display_name = str(name)
        print(f"   • {display_name}")
    if len(risk_patterns) > 5:
        print(f"   ... and {len(risk_patterns) - 5} more")
    
    results = []
    
    if args.clause:
        print(f"\n" + "="*70)
        print("MODE: Single Clause Analysis")
        print("="*70)
        
        prediction = predict_single_clause(model, tokenizer, args.clause, config)
        print(format_prediction_output(prediction, risk_patterns))
        results.append(prediction)
    
    elif args.document:
        print(f"\n" + "="*70)
        print("MODE: Full Document Analysis (with context)")
        print("="*70)
        
        with open(args.document, 'r') as f:
            doc_data = json.load(f)
        
        document = doc_data.get('sections', [])
        
        prediction = predict_document(model, tokenizer, document, config)
        
        print(f"\n Document Summary:")
        print(f"   Sections: {prediction['summary']['num_sections']}")
        print(f"   Clauses: {prediction['summary']['num_clauses']}")
        print(f"   Average Severity: {prediction['summary']['avg_severity']:.2f}/10")
        print(f"   High Risk Clauses: {prediction['summary']['high_risk_count']}")
        
        print(f"\n Clause-by-Clause Analysis:")
        for clause_pred in prediction['clauses']:
            print(format_prediction_output(clause_pred, risk_patterns))
        
        results = prediction
    
    else:
        print(f"\n" + "="*70)
        print("MODE: Demo Analysis")
        print("="*70)
        print("\n Running demo with sample clauses...")
        
        demo_clauses = [
            "The party shall indemnify and hold harmless all damages and losses.",
            "This agreement shall be governed by the laws of the state of California.",
            "Payment must be made within thirty days of invoice date.",
            "The licensee must not disclose confidential information to third parties.",
            "Company shall comply with all applicable laws and regulations."
        ]
        
        for clause in demo_clauses:
            prediction = predict_single_clause(model, tokenizer, clause, config)
            print(format_prediction_output(prediction, risk_patterns))
            results.append(prediction)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n Results saved to: {args.output}")
    
    print("\n" + "="*70)
    print(" INFERENCE COMPLETE")
    print("="*70)
    
    if not args.clause and not args.document:
        print(f"\n Usage Examples:")
        print(f'\n   Single clause:')
        print(f'   python3 inference.py --clause "The party shall indemnify..."')
        print(f'\n   Full document:')
        print(f'   python3 inference.py --document contract.json')
        print(f'\n   Save results:')
        print(f'   python3 inference.py --clause "..." --output results.json')


if __name__ == "__main__":
    main()
