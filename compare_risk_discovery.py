import argparse
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Union
import time

from data_loader import CUADDataLoader
from risk_discovery import UnsupervisedRiskDiscovery
from risk_discovery_alternatives import (
    TopicModelingRiskDiscovery,
    HierarchicalRiskDiscovery,
    DensityBasedRiskDiscovery,
    NMFRiskDiscovery,
    SpectralClusteringRiskDiscovery,
    GaussianMixtureRiskDiscovery,
    MiniBatchKMeansRiskDiscovery,
    compare_risk_discovery_methods
)
from risk_o_meter import RiskOMeterFramework


def load_sample_data(data_path: str, max_clauses: Union[int, None] = 5000) -> List[str]:
    print(f" Loading CUAD dataset from {data_path}...")
    
    try:
        data_loader = CUADDataLoader(data_path)
        all_data = data_loader.load_data()
        clauses: List[str] = []

        if isinstance(all_data, tuple) and all_data:
            df_candidate = all_data[0]
            try:
                if hasattr(df_candidate, '__getitem__') and 'clause_text' in df_candidate:
                    clauses.extend([str(text) for text in df_candidate['clause_text'].tolist()])
            except Exception:
                pass

        if not clauses:
            for item in all_data:
                if isinstance(item, dict) and 'clause_text' in item:
                    clauses.append(str(item['clause_text']))
                elif isinstance(item, str):
                    clauses.append(item)

        print(f"  Loaded {len(clauses)} clauses before limiting")

        if max_clauses is not None and len(clauses) > max_clauses:
            print(f"  Using {max_clauses} out of {len(clauses)} clauses for comparison")
            clauses = clauses[:max_clauses]
        else:
            print("  Using full dataset")
        
        return clauses
    
    except Exception as e:
        print(f" Could not load data: {e}")
        print("  Using synthetic sample data for demonstration")
        return generate_sample_clauses()


def generate_sample_clauses() -> List[str]:
    sample_clauses = [
        "The Company shall not be liable for any indirect, incidental, or consequential damages arising from use of the services.",
        "Licensor's total liability under this Agreement shall not exceed the fees paid in the twelve months preceding the claim.",
        "In no event shall either party be liable for any loss of profits, business interruption, or loss of data.",
        
        "The Service Provider agrees to indemnify and hold harmless the Client from any claims arising from breach of this Agreement.",
        "Customer shall indemnify Company against all third-party claims related to Customer's use of the Software.",
        "Each party shall indemnify the other for losses resulting from the indemnifying party's gross negligence or willful misconduct.",
        
        "Either party may terminate this Agreement upon thirty (30) days written notice to the other party.",
        "This Agreement shall automatically terminate if either party files for bankruptcy or becomes insolvent.",
        "Upon termination, Customer must immediately cease use of the Software and destroy all copies.",
        
        "All intellectual property rights in the deliverables shall remain the exclusive property of the Company.",
        "Customer grants Vendor a non-exclusive license to use Customer's trademarks solely for providing the services.",
        "Any modifications or derivative works created by Licensor shall be owned by Licensor.",
        
        "Each party shall keep confidential all information disclosed by the other party marked as 'Confidential'.",
        "The obligation of confidentiality shall survive termination of this Agreement for a period of five (5) years.",
        "Confidential Information does not include information that is publicly available or independently developed.",
        
        "Customer agrees to pay the monthly subscription fee of $10,000 within 15 days of invoice.",
        "All fees are non-refundable and must be paid in U.S. dollars.",
        "Late payments shall accrue interest at the rate of 1.5% per month or the maximum allowed by law.",
        
        "Both parties agree to comply with all applicable federal, state, and local laws and regulations.",
        "Vendor shall maintain compliance with SOC 2 Type II and ISO 27001 standards.",
        "Customer is responsible for ensuring its use of the Services complies with GDPR and other data protection laws.",
        
        "Company warrants that the Software will perform substantially in accordance with the documentation.",
        "Vendor represents and warrants that it has the right to enter into this Agreement and grant the licenses herein.",
        "EXCEPT AS EXPRESSLY PROVIDED, THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT WARRANTY OF ANY KIND.",
    ]

    clauses = sample_clauses * 50
    print(f"  Generated {len(clauses)} sample clauses for demonstration")
    
    return clauses


def compare_single_method(method_name: str, discovery_object, clauses: List[str], 
                         n_patterns: int = 7) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"Testing: {method_name}")
    print(f"{'='*80}")
    start_time = time.time()
    
    try:
        results = discovery_object.discover_risk_patterns(clauses)
        elapsed_time = time.time() - start_time
        
        print(f"\n  Execution time: {elapsed_time:.2f} seconds")

        results['execution_time'] = elapsed_time
        results['clauses_per_second'] = len(clauses) / elapsed_time
        
        return {
            'success': True,
            'results': results,
            'execution_time': elapsed_time
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f" Error: {e}")
        
        return {
            'success': False,
            'error': str(e),
            'execution_time': elapsed_time
        }


def analyze_pattern_diversity(results: Dict[str, Any]) -> Dict[str, float]:
    metrics = {}
    
    if 'discovered_topics' in results:
        patterns = results['discovered_topics']
        sizes = [p['clause_count'] for p in patterns.values()]
    elif 'discovered_clusters' in results:
        patterns = results['discovered_clusters']
        sizes = [p['clause_count'] for p in patterns.values()]
    elif 'discovered_patterns' in results:
        patterns = results['discovered_patterns']
        sizes = [p.get('clause_count', p.get('size', 0)) for p in patterns.values()]
    else:
        return metrics
    
    if sizes:
        metrics['avg_pattern_size'] = float(np.mean(sizes))
        metrics['std_pattern_size'] = float(np.std(sizes))
        metrics['min_pattern_size'] = int(np.min(sizes))
        metrics['max_pattern_size'] = int(np.max(sizes))

        cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
        metrics['balance_score'] = float(1.0 / (1.0 + cv))
    
    return metrics


def generate_comparison_report(all_results: Dict[str, Dict]) -> str:
    report = []
    report.append("=" * 80)
    report.append(" RISK DISCOVERY METHOD COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append(" SUMMARY TABLE")
    report.append("-" * 80)
    report.append(f"{'Method':<30} {'Patterns':<12} {'Quality':<20}")
    report.append("-" * 80)
    
    for method_name, result in all_results.items():
        n_patterns = result.get('n_clusters') or result.get('n_topics') or result.get('n_components', 'N/A')
        
        quality_metrics = result.get('quality_metrics', {})
        if 'silhouette_score' in quality_metrics:
            sil_score = quality_metrics['silhouette_score']
            if isinstance(sil_score, (int, float)):
                quality = f"Silhouette: {sil_score:.3f}"
            else:
                quality = f"Silhouette: {sil_score}"
        elif 'perplexity' in quality_metrics:
            perp = quality_metrics['perplexity']
            if isinstance(perp, (int, float)):
                quality = f"Perplexity: {perp:.1f}"
            else:
                quality = f"Perplexity: {perp}"
        else:
            quality = "See details"
        
        report.append(f"{method_name:<30} {str(n_patterns):<12} {quality:<20}")
    
    report.append("-" * 80)
    report.append("")
    
    report.append(" DETAILED ANALYSIS")
    report.append("=" * 80)
    
    for method_name, result in all_results.items():
        report.append(f"\n{method_name.upper()}")
        report.append("-" * 80)

        report.append(f"Method: {result.get('method', 'Unknown')}")

        n_patterns = result.get('n_clusters') or result.get('n_topics') or result.get('n_components', 0)
        report.append(f"Patterns Discovered: {n_patterns}")

        if 'quality_metrics' in result:
            report.append("Quality Metrics:")
            for metric, value in result['quality_metrics'].items():
                if isinstance(value, float):
                    report.append(f"  - {metric}: {value:.3f}")
                else:
                    report.append(f"  - {metric}: {value}")

        diversity = analyze_pattern_diversity(result)
        if diversity:
            report.append("Pattern Diversity:")
            for metric, value in diversity.items():
                report.append(f"  - {metric}: {value:.3f}" if isinstance(value, float) else f"  - {metric}: {value}")

        if 'discovered_topics' in result:
            report.append("\nTop 3 Topics:")
            for i, (topic_id, topic) in enumerate(list(result['discovered_topics'].items())[:3]):
                report.append(f"  Topic {topic_id}: {topic['topic_name']}")
                report.append(f"    Keywords: {', '.join(topic['top_words'][:5])}")
                report.append(f"    Clauses: {topic['clause_count']} ({topic['proportion']:.1%})")
        
        elif 'discovered_clusters' in result:
            report.append("\nTop 3 Clusters:")
            for i, (cluster_id, cluster) in enumerate(list(result['discovered_clusters'].items())[:3]):
                report.append(f"  Cluster {cluster_id}: {cluster['cluster_name']}")
                report.append(f"    Keywords: {', '.join(cluster['top_terms'][:5])}")
                report.append(f"    Clauses: {cluster['clause_count']} ({cluster['proportion']:.1%})")
        
        elif 'discovered_patterns' in result:
            report.append("\nTop 3 Patterns:")
            for i, (pattern_id, pattern) in enumerate(list(result['discovered_patterns'].items())[:3]):
                pattern_name = pattern_id if isinstance(pattern_id, str) else pattern.get('name', f'Pattern {pattern_id}')
                keywords = pattern.get('key_terms', pattern.get('top_keywords', []))
                clause_count = pattern.get('clause_count', pattern.get('size', 0))
                
                report.append(f"  {pattern_name}")
                if keywords:
                    report.append(f"    Keywords: {', '.join(keywords[:5])}")
                report.append(f"    Clauses: {clause_count}")

        if method_name == 'dbscan' and 'n_outliers' in result:
            report.append(f"\nOutliers Detected: {result['n_outliers']} ({result['quality_metrics'].get('outlier_ratio', 0):.1%})")
            report.append("  → These represent rare or unique risk patterns")
    
    report.append("\n" + "=" * 80)
    report.append(" RECOMMENDATIONS BY METHOD")
    report.append("=" * 80)
    
    report.append("""
═══ BASIC METHODS (Fast & Reliable) ═══

1. K-MEANS (Original):
    Best for: Fast, scalable clustering with clear boundaries
    Use when: You need consistent performance and interpretability
    Speed: Very Fast |  Accuracy: Good |  Scalability: Excellent
   
2. LDA TOPIC MODELING:
    Best for: Discovering overlapping risk categories
    Use when: Clauses may belong to multiple risk types
    Speed: Moderate |  Accuracy: Very Good |  Scalability: Good
   
3. HIERARCHICAL CLUSTERING:
    Best for: Understanding risk relationships and hierarchies
    Use when: You want to explore risk structure at different levels
    Speed: Moderate |  Accuracy: Good |  Scalability: Limited (<10K clauses)
   
4. DBSCAN:
    Best for: Finding rare/unusual risks and handling outliers
    Use when: You need to identify unique risk patterns
    Speed: Fast |  Accuracy: Good |  Scalability: Good

═══ ADVANCED METHODS (Comprehensive Analysis) ═══

5. NMF (Non-negative Matrix Factorization):
    Best for: Parts-based decomposition with interpretable components
    Use when: You want additive risk factors (clause = sum of components)
    Speed: Fast |  Accuracy: Very Good |  Scalability: Excellent
    Unique: Components are non-negative, highly interpretable
   
6. SPECTRAL CLUSTERING:
    Best for: Complex relationships and non-convex cluster shapes
    Use when: Risk patterns have intricate graph-like relationships
    Speed: Slow |  Accuracy: Excellent |  Scalability: Limited (<5K clauses)
    Unique: Uses eigenvalue decomposition, best quality for small datasets
   
7. GAUSSIAN MIXTURE MODEL:
    Best for: Soft probabilistic clustering with uncertainty estimates
    Use when: You need confidence scores for risk assignments
    Speed: Moderate |  Accuracy: Very Good |  Scalability: Good
    Unique: Provides probability distributions, quantifies uncertainty
   
8. MINI-BATCH K-MEANS:
    Best for: Ultra-large datasets (100K+ clauses)
    Use when: You need K-Means quality at 3-5x faster speed
    Speed: Ultra Fast |  Accuracy: Good |  Scalability: Extreme (>1M clauses)
    Unique: Online learning, extremely memory efficient

9. RISK-O-METER (Doc2Vec + SVM)  PAPER BASELINE:
    Best for: Supervised learning with labeled data
    Use when: You have risk labels and want paper-validated approach
    Speed: Moderate |  Accuracy: Excellent (91% reported) |  Scalability: Good
    Unique: Paragraph vectors capture semantic meaning, proven in literature
   Reference: Chakrabarti et al., 2018 - "Risk-o-meter framework"

═══ SELECTION GUIDE ═══

 Dataset Size:
   • <1K clauses: Use Spectral or GMM for best quality
   • 1K-10K clauses: All methods work well
   • 10K-100K clauses: Avoid Hierarchical and Spectral
   • >100K clauses: Use Mini-Batch K-Means

 Quality Priority:
   • Highest: Spectral, GMM, LDA
   • Balanced: NMF, K-Means
   • Speed-focused: Mini-Batch, DBSCAN

 Special Requirements:
   • Overlapping risks: LDA, GMM
   • Outlier detection: DBSCAN
   • Hierarchical structure: Hierarchical
   • Interpretability: NMF, LDA
   • Uncertainty estimates: GMM, LDA
""")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare risk discovery methods on CUAD dataset")
    parser.add_argument("--advanced", "-a", action="store_true", help="Include advanced methods in comparison")
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=None,
        help="Maximum number of clauses to use (omit for full dataset)"
    )
    parser.add_argument(
        "--data-path",
        default="dataset/CUAD_v1/CUAD_v1.json",
        help="Path to CUAD dataset JSON file"
    )
    return parser.parse_args()


def main():
    print("=" * 80)
    args = parse_args()

    include_advanced = args.advanced
    
    print(" RISK DISCOVERY METHOD COMPARISON")
    print("=" * 80)
    print("")
    if include_advanced:
        print(" FULL COMPARISON MODE (9 Methods)")
        print("")
        print("BASIC METHODS:")
        print("  1. K-Means Clustering")
        print("  2. LDA Topic Modeling")
        print("  3. Hierarchical Clustering")
        print("  4. DBSCAN (Density-Based)")
        print("")
        print("ADVANCED METHODS:")
        print("  5. NMF (Matrix Factorization)")
        print("  6. Spectral Clustering")
        print("  7. Gaussian Mixture Model")
        print("  8. Mini-Batch K-Means")
        print("  9. Risk-o-meter (Doc2Vec + SVM) PAPER BASELINE")
    else:
        print(" QUICK COMPARISON MODE (4 Basic Methods)")
        print("")
        print("  1. K-Means Clustering (Original)")
        print("  2. LDA Topic Modeling")
        print("  3. Hierarchical Clustering")
        print("  4. DBSCAN (Density-Based)")
        print("")
        print(" Tip: Use --advanced flag for all 9 methods")
    print("")
    
    clauses = load_sample_data(args.data_path, max_clauses=args.max_clauses)
    
    if not clauses:
        print(" No clauses loaded. Exiting.")
        return
    
    print(f"\n Loaded {len(clauses)} clauses for comparison")
    
    n_patterns = 7
    
    print("\n" + "=" * 80)
    print(" RUNNING UNIFIED COMPARISON")
    print("=" * 80)
    
    start_time = time.time()
    comparison_results = compare_risk_discovery_methods(
        clauses, 
        n_patterns=n_patterns,
        include_advanced=include_advanced
    )
    total_time = time.time() - start_time
    
    all_results = comparison_results['detailed_results']
    summary = comparison_results['summary']
    
    print(f"\n  Total Comparison Time: {total_time:.2f} seconds")
    
    print("\n" + "=" * 80)
    print(" GENERATING COMPARISON REPORT")
    print("=" * 80)
    
    report = generate_comparison_report(all_results)
    print("\n" + report)
    
    print("\n" + "=" * 80)
    print(" SAVING RESULTS")
    print("=" * 80)
    
    with open('risk_discovery_comparison_report.txt', 'w') as f:
        f.write(report)
    print(" Report saved to: risk_discovery_comparison_report.txt")
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {
                (str(k) if isinstance(k, (np.integer, np.floating)) else k): convert_for_json(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(all_results)
    with open('risk_discovery_comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(" Detailed results saved to: risk_discovery_comparison_results.json")
    
    print("\n" + "=" * 80)
    print(" COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
