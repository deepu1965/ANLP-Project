import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re
from sklearn.model_selection import train_test_split

class CUADDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df_clauses = None
        self.contracts = None
        self.splits = None
        
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and parse CUAD dataset"""
        print(f" Loading CUAD dataset from {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            cuad_data = json.load(f)
        
        clauses_data = []
        
        for item in cuad_data['data']:
            title = item['title']
            
            for paragraph in item['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    question = qa['question']
                    clause_category = question

                    for answer in qa['answers']:
                        clause_text = answer['text']
                        start_pos = answer['answer_start']
                        
                        clauses_data.append({
                            'filename': title,
                            'clause_text': clause_text,
                            'category': clause_category,
                            'start_position': start_pos,
                            'contract_context': context
                        })
        
        self.df_clauses = pd.DataFrame(clauses_data)
        
        self.contracts = self.df_clauses.groupby('filename').agg({
            'clause_text': list,
            'category': list,
            'contract_context': 'first'
        }).reset_index()
        
        print(f" Loaded {len(self.df_clauses)} clauses from {len(self.contracts)} contracts")
        print(f" Found {self.df_clauses['category'].nunique()} unique clause categories")
        
        return self.df_clauses, self.contracts.set_index('filename').to_dict('index')
    
    def create_splits(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        if self.contracts is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        unique_contracts = self.contracts['filename'].unique()
        
        train_val_contracts, test_contracts = train_test_split(
            unique_contracts,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        train_contracts, val_contracts = train_test_split(
            train_val_contracts,
            test_size=val_size/(1-test_size),
            random_state=random_state,
            shuffle=True
        )
        
        train_clauses = self.df_clauses[self.df_clauses['filename'].isin(train_contracts)]
        val_clauses = self.df_clauses[self.df_clauses['filename'].isin(val_contracts)]
        test_clauses = self.df_clauses[self.df_clauses['filename'].isin(test_contracts)]
        
        self.splits = {
            'train': train_clauses,
            'val': val_clauses,
            'test': test_clauses
        }
        
        print(f" Data splits created:")
        print(f"  Train: {len(train_clauses)} clauses from {len(train_contracts)} contracts")
        print(f"  Val: {len(val_clauses)} clauses from {len(val_contracts)} contracts")
        print(f"  Test: {len(test_clauses)} clauses from {len(test_contracts)} contracts")
        
        return self.splits
    
    def get_clause_texts(self, split: str = 'train') -> List[str]:
        if self.splits is None:
            raise ValueError("Splits must be created first using create_splits()")
        
        return self.splits[split]['clause_text'].tolist()
    
    def get_categories(self, split: str = 'train') -> List[str]:
        if self.splits is None:
            raise ValueError("Splits must be created first using create_splits()")
        
        return self.splits[split]['category'].tolist()
    
    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\w\s.,;:()"-]', ' ', text)
        
        text = text.strip()
        
        return text

class ContractDataPipeline:
    def __init__(self):
        self.clause_boundary_patterns = [
            r'\n\s*\d+\.\s+',
            r'\n\s*\([a-zA-Z0-9]+\)\s+',
            r'\n\s*[A-Z][A-Z\s]{10,}:',
            r'\.\s+[A-Z][a-z]+\s+shall',
            r'\.\s+[A-Z][a-z]+\s+agrees?',
            r'\.\s+In\s+the\s+event\s+that',
        ]
        
        self.entity_patterns = {
            'monetary': r'\$[\d,]+(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%',
            'time_period': r'\d+\s*(?:days?|months?|years?|weeks?)',
            'legal_entities': r'(?:Inc\.|LLC|Corp\.|Corporation|Company|Ltd\.)',
            'parties': r'\b(?:Party|Parties|Company|Corporation|Licensor|Licensee|Vendor|Customer)\b',
            'dates': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        }
        
        self.complexity_indicators = {
            'modal_verbs': r'\b(?:shall|must|may|should|will|might|could|would)\b',
            'conditional_terms': r'\b(?:if|unless|provided|subject to|in the event|notwithstanding)\b',
            'legal_conjunctions': r'\b(?:whereas|therefore|furthermore|moreover|however)\b',
            'obligation_terms': r'\b(?:agrees?|undertakes?|covenants?|warrants?|represents?)\b'
        }
    
    def clean_clause_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\w\s\.\,\;\:\(\)\-\"\'\$\%]', ' ', text)
        
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        return text.strip()
    
    def extract_legal_entities(self, text: str) -> Dict:
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = matches
        
        return entities
    
    def calculate_text_complexity(self, text: str) -> float:
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        features = {
            'avg_word_length': sum(len(word) for word in words) / len(words),
            'long_words': sum(1 for word in words if len(word) > 6) / len(words),
            'sentences': len(re.split(r'[.!?]+', text)),
            'subordinate_clauses': (text.count(',') + text.count(';')) / len(words) * 100,
        }
        
        for indicator_type, pattern in self.complexity_indicators.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            features[indicator_type] = matches / len(words) * 100
        
        complexity = (
            min(features['avg_word_length'] / 8, 1) * 2 +
            features['long_words'] * 2 +
            min(features['subordinate_clauses'] / 5, 1) * 2 +
            min(features['conditional_terms'] / 2, 1) * 2 +
            min(features['modal_verbs'] / 3, 1) * 2
        )
        
        return min(complexity, 10)
    
    def prepare_clause_for_bert(self, clause_text: str, max_length: int = 512) -> Dict:
        clean_text = self.clean_clause_text(clause_text)
        
        words = clean_text.split()
        
        if len(words) > max_length - 10:
            words = words[:max_length-10]
            clean_text = ' '.join(words)
            truncated = True
        else:
            truncated = False
        
        entities = self.extract_legal_entities(clean_text)
        
        return {
            'text': clean_text,
            'word_count': len(words),
            'char_count': len(clean_text),
            'sentence_count': len(re.split(r'[.!?]+', clean_text)),
            'truncated': truncated,
            'entities': entities,
            'complexity_score': self.calculate_text_complexity(clean_text)
        }
    
    def process_clauses(self, df_clauses: pd.DataFrame) -> pd.DataFrame:
        print(f" Processing {len(df_clauses)} clauses through data pipeline...")
        
        processed_data = []
        total_clauses = len(df_clauses)
        
        for idx, row in df_clauses.iterrows():
            if idx % 1000 == 0 and idx > 0:
                print(f"  Processed {idx}/{total_clauses} clauses ({(idx/total_clauses)*100:.1f}%)")
            
            bert_ready = self.prepare_clause_for_bert(row['clause_text'])
            
            processed_data.append({
                'filename': row['filename'],
                'category': row['category'],
                'original_text': row['clause_text'],
                'processed_text': bert_ready['text'],
                'word_count': bert_ready['word_count'],
                'char_count': bert_ready['char_count'],
                'sentence_count': bert_ready['sentence_count'],
                'truncated': bert_ready['truncated'],
                'complexity_score': bert_ready['complexity_score'],
                'monetary_amounts': len(bert_ready['entities']['monetary']),
                'time_periods': len(bert_ready['entities']['time_period']),
                'legal_entities': len(bert_ready['entities']['legal_entities']),
            })
        
        print(f" Completed processing {total_clauses} clauses")
        return pd.DataFrame(processed_data)
