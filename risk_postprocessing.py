import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import re


def merge_duplicate_topics(discovered_patterns: Dict, cluster_labels: np.ndarray, 
                           merge_rules: Dict[str, List[str]] = None) -> tuple:
    
    if 'discovered_topics' in discovered_patterns:
        topics = discovered_patterns['discovered_topics']
    else:
        topics = discovered_patterns
    
    if merge_rules is None:
        merge_rules = detect_duplicate_topics(discovered_patterns)
    
    if not merge_rules:
        print("  No duplicate topics detected - no merging needed")
        return topics, cluster_labels
    
    print(f" Merging duplicate topics...")
    
    old_to_new = {}
    new_id = 0
    merged_patterns = {}
    merged_old_ids = set()
    
    for new_name, old_names_or_ids in merge_rules.items():
        print(f"   Merging {len(old_names_or_ids)} topics → {new_name}")
        
        patterns_to_merge = []
        old_ids_to_merge = []
        
        for old_ref in old_names_or_ids:
            if isinstance(old_ref, int):
                old_id = old_ref
                old_ids_to_merge.append(old_id)
            else:
                for pattern_id, pattern in topics.items():
                    pattern_name = pattern.get('topic_name') or pattern.get('pattern_name', '')
                    if old_ref in pattern_name or pattern_name in old_ref:
                        old_id = int(pattern_id) if isinstance(pattern_id, str) and pattern_id.isdigit() else pattern_id
                        old_ids_to_merge.append(old_id)
            pattern_key = str(old_id) if isinstance(old_id, int) else old_id
            if pattern_key in topics:
                patterns_to_merge.append(topics[pattern_key])
                merged_old_ids.add(pattern_key)
        
        if patterns_to_merge:
            merged_pattern = merge_topic_data(patterns_to_merge, new_name)
            merged_patterns[str(new_id)] = merged_pattern
            for old_id in old_ids_to_merge:
                old_to_new[old_id] = new_id
            
            new_id += 1
    
    for pattern_id, pattern in topics.items():
        if pattern_id not in merged_old_ids:
            old_id = int(pattern_id) if isinstance(pattern_id, str) and pattern_id.isdigit() else pattern_id
            old_to_new[old_id] = new_id
            merged_patterns[str(new_id)] = pattern.copy()
            merged_patterns[str(new_id)]['topic_id'] = new_id
            new_id += 1
    
    new_labels = np.array([old_to_new.get(label, label) for label in cluster_labels])
    
    print(f" Merging complete: {len(discovered_patterns)} → {len(merged_patterns)} topics")
    
    return merged_patterns, new_labels


def detect_duplicate_topics(discovered_patterns: Dict) -> Dict[str, List]:
    
    merge_rules = {}
    if 'discovered_topics' in discovered_patterns:
        topics = discovered_patterns['discovered_topics']
    else:
        topics = discovered_patterns
    
    base_name_groups = defaultdict(list)
    
    for topic_id, topic in topics.items():
        topic_name = topic.get('topic_name') or topic.get('pattern_name', '')
        
        base_name = re.sub(r'[(_\s].+', '', topic_name).upper()
        base_name = base_name.replace('TOPIC_', '').replace('PATTERN_', '')
        
        if base_name:
            topic_id_int = int(topic_id) if isinstance(topic_id, str) and topic_id.isdigit() else topic_id
            base_name_groups[base_name].append(topic_id_int)
    
    for base_name, topic_ids in base_name_groups.items():
        if len(topic_ids) > 1:
            merge_rules[base_name] = topic_ids
            print(f"    Detected duplicate: {len(topic_ids)} topics with base name '{base_name}'")
    
    return merge_rules


def merge_topic_data(patterns: List[Dict], new_name: str) -> Dict:
    merged = {
        'topic_name': f"Topic_{new_name}",
        'clause_count': sum(p.get('clause_count', 0) for p in patterns),
    }
    
    all_keywords = []
    for pattern in patterns:
        keywords = pattern.get('keywords', pattern.get('top_words', []))
        all_keywords.extend(keywords[:10])
    
    from collections import Counter
    keyword_counts = Counter(all_keywords)
    merged['top_words'] = [word for word, _ in keyword_counts.most_common(15)]
    merged['keywords'] = merged['top_words']
    
    if 'word_weights' in patterns[0]:
        all_weights = []
        for pattern in patterns:
            weights = pattern.get('word_weights', [])
            all_weights.extend(weights[:10])
        merged['word_weights'] = sorted(all_weights, reverse=True)[:15]
    
    numeric_fields = ['avg_risk_intensity', 'avg_legal_complexity', 'avg_obligation_strength', 'proportion']
    for field in numeric_fields:
        values = [p.get(field, 0) for p in patterns if field in p]
        if values:
            merged[field] = np.mean(values)
    
    all_samples = []
    for pattern in patterns:
        samples = pattern.get('sample_clauses', [])
        all_samples.extend(samples[:2])
    merged['sample_clauses'] = all_samples[:5]
    
    return merged


def validate_cluster_quality(discovered_patterns: Dict, min_cluster_size: int = 150) -> Dict:
    report = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'cluster_sizes': {}
    }
    
    if 'discovered_topics' in discovered_patterns:
        topics = discovered_patterns['discovered_topics']
    elif any(isinstance(v, dict) and ('topic_name' in v or 'pattern_name' in v or 'key_terms' in v) 
             for v in discovered_patterns.values()):
        topics = discovered_patterns
    else:
        report['is_valid'] = False
        report['issues'].append("Invalid format: expected 'discovered_topics' key or topics dictionary")
        return report
    
    sizes = []
    names = []
    
    for topic_id, topic in topics.items():
        count = topic.get('clause_count', 0)
        name = topic.get('topic_name', topic.get('pattern_name', f"Topic_{topic_id}"))
        
        sizes.append(count)
        names.append(name)
        report['cluster_sizes'][name] = count
        
        if count < min_cluster_size:
            report['is_valid'] = False
            report['issues'].append(f"Cluster '{name}' too small: {count} < {min_cluster_size}")
    
    from collections import Counter
    name_counts = Counter(names)
    for name, count in name_counts.items():
        if count > 1:
            report['is_valid'] = False
            report['issues'].append(f"Duplicate cluster name: '{name}' appears {count} times")
    
    if sizes:
        max_size = max(sizes)
        min_size = min(sizes)
        ratio = max_size / min_size if min_size > 0 else float('inf')
        
        if ratio > 3.0:
            report['warnings'].append(
                f"Imbalanced clusters: largest ({max_size}) is {ratio:.1f}x bigger than smallest ({min_size})"
            )
    
    return report


if __name__ == "__main__":
    print(" Risk Discovery Post-Processing Utilities\n")
    
    test_patterns = {
        '0': {'topic_name': 'Topic_LIABILITY', 'clause_count': 400, 'top_words': ['insurance', 'coverage']},
        '1': {'topic_name': 'Topic_COMPLIANCE', 'clause_count': 300, 'top_words': ['laws', 'governed']},
        '2': {'topic_name': 'Topic_TERMINATION', 'clause_count': 350, 'top_words': ['term', 'notice']},
        '6': {'topic_name': 'Topic_LIABILITY', 'clause_count': 250, 'top_words': ['damages', 'breach']},
    }
    
    test_labels = np.array([0, 1, 2, 0, 1, 6, 2, 0, 6])
    
    print("1. Detecting duplicate topics:")
    merge_rules = detect_duplicate_topics(test_patterns)
    print()
    
    print("2. Merging duplicates:")
    merged_patterns, new_labels = merge_duplicate_topics(test_patterns, test_labels, merge_rules)
    print()
    
    print("3. Validating cluster quality:")
    report = validate_cluster_quality(merged_patterns, min_cluster_size=200)
    print(f"   Valid: {report['is_valid']}")
    print(f"   Issues: {report['issues']}")
    print(f"   Warnings: {report['warnings']}")
