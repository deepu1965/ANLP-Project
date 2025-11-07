import re
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import warnings


class TopicModelingRiskDiscovery:
    def __init__(self, n_topics: int = 7, random_state: int = 42):
        self.n_topics = n_topics
        self.random_state = random_state
        
        self.vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=3,
            max_df=0.85
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=20,
            learning_method='batch',
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
            n_jobs=-1
        )
        
        self.discovered_topics = {}
        self.topic_labels = None
        self.feature_matrix = None
        self.topic_word_distribution = None
        
    def discover_risk_patterns(self, clauses: List[str]) -> Dict[str, Any]:
        print(f" Discovering risk topics using LDA (n_topics={self.n_topics})...")
        cleaned_clauses = [self._clean_text(c) for c in clauses]
        print("   Creating document-term matrix...")
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_clauses)
        feature_names = self.vectorizer.get_feature_names_out()
        print("   Fitting LDA model...")
        self.lda_model.fit(self.feature_matrix)
        self.topic_word_distribution = self.lda_model.components_
        doc_topic_dist = self.lda_model.transform(self.feature_matrix)
        self.topic_labels = np.argmax(doc_topic_dist, axis=1)
        print("   Extracting topic keywords...")
        n_top_words = 15
        for topic_idx in range(self.n_topics):
            top_word_indices = np.argsort(self.topic_word_distribution[topic_idx])[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            top_weights = [self.topic_word_distribution[topic_idx][i] for i in top_word_indices]
            topic_name = self._generate_topic_name(top_words)
            clause_count = np.sum(self.topic_labels == topic_idx)
            
            self.discovered_topics[topic_idx] = {
                'topic_id': topic_idx,
                'topic_name': topic_name,
                'top_words': top_words,
                'word_weights': top_weights,
                'clause_count': int(clause_count),
                'proportion': float(clause_count / len(clauses))
            }
        
        perplexity = self.lda_model.perplexity(self.feature_matrix)
        log_likelihood = self.lda_model.score(self.feature_matrix)
        
        print(f" LDA discovery complete: {self.n_topics} topics found")
        print(f"   Perplexity: {perplexity:.2f} (lower is better)")
        print(f"   Log-likelihood: {log_likelihood:.2f}")
        
        return {
            'method': 'LDA_Topic_Modeling',
            'n_topics': self.n_topics,
            'discovered_topics': self.discovered_topics,
            'topic_labels': self.topic_labels,
            'doc_topic_distribution': doc_topic_dist,
            'perplexity': perplexity,
            'log_likelihood': log_likelihood,
            'quality_metrics': {
                'perplexity': perplexity,
                'avg_topic_diversity': self._compute_topic_diversity()
            }
        }
    
    def get_clause_topic_distribution(self, clause_idx: int) -> Dict[int, float]:
        if self.feature_matrix is None:
            return {}
        
        doc_topic_dist = self.lda_model.transform(self.feature_matrix)
        return {topic_id: float(prob) for topic_id, prob in enumerate(doc_topic_dist[clause_idx])}
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _generate_topic_name(self, top_words: List[str]) -> str:
        themes = {
            'liability': ['liability', 'liable', 'damages', 'loss', 'harm', 'injury'],
            'indemnity': ['indemnify', 'indemnification', 'hold', 'harmless', 'defend'],
            'termination': ['terminate', 'termination', 'cancel', 'end', 'expire'],
            'intellectual_property': ['intellectual', 'property', 'ip', 'patent', 'copyright', 'trademark'],
            'confidentiality': ['confidential', 'confidentiality', 'disclosure', 'nda', 'secret'],
            'payment': ['payment', 'pay', 'fee', 'price', 'cost', 'charge'],
            'compliance': ['comply', 'compliance', 'regulation', 'law', 'legal', 'regulatory'],
            'warranty': ['warranty', 'warrant', 'represent', 'guarantee', 'assure']
        }
        theme_scores = defaultdict(int)
        for word in top_words[:10]:
            for theme, keywords in themes.items():
                if any(keyword in word.lower() for keyword in keywords):
                    theme_scores[theme] += 1
        if theme_scores:
            best_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
            return f"Topic_{best_theme.upper()}"
        else:
            return f"Topic_{top_words[0].upper()}_{top_words[1].upper()}"
    
    def _compute_topic_diversity(self) -> float:
        diversities = []
        for topic_idx in range(self.n_topics):
            word_dist = self.topic_word_distribution[topic_idx]
            word_dist = word_dist / np.sum(word_dist)
            entropy = -np.sum(word_dist * np.log(word_dist + 1e-10))
            diversities.append(entropy)
        return float(np.mean(diversities))


class HierarchicalRiskDiscovery:
    def __init__(self, n_clusters: int = 7, linkage: str = 'ward', random_state: int = 42):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.90
        )
        self.clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        self.discovered_clusters = {}
        self.cluster_labels = None
        self.feature_matrix = None
        
    def discover_risk_patterns(self, clauses: List[str]) -> Dict[str, Any]:
        print(f" Discovering risk patterns using Hierarchical Clustering (n_clusters={self.n_clusters})...")
        cleaned_clauses = [self._clean_text(c) for c in clauses]
        print("   Creating TF-IDF feature matrix...")
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_clauses)
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"   Fitting Hierarchical Clustering (linkage={self.linkage})...")
        self.cluster_labels = self.clustering_model.fit_predict(self.feature_matrix.toarray())
        print("   Analyzing discovered clusters...")
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_clauses = [clauses[i] for i in cluster_indices]
            cluster_tfidf = self.feature_matrix[cluster_mask].mean(axis=0)
            top_term_indices = np.argsort(np.asarray(cluster_tfidf).flatten())[-15:][::-1]
            top_terms = [feature_names[i] for i in top_term_indices]
            top_scores = [float(cluster_tfidf[0, i]) for i in top_term_indices]
            cluster_name = self._generate_cluster_name(top_terms)
            self.discovered_clusters[cluster_id] = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'top_terms': top_terms,
                'term_scores': top_scores,
                'clause_count': int(len(cluster_indices)),
                'proportion': float(len(cluster_indices) / len(clauses)),
                'sample_clauses': cluster_clauses[:3]
            }
        if len(clauses) < 10000:
            silhouette = silhouette_score(self.feature_matrix, self.cluster_labels)
        else:
            silhouette = None
        print(f" Hierarchical clustering complete: {self.n_clusters} clusters found")
        if silhouette:
            print(f"   Silhouette Score: {silhouette:.3f} (range: -1 to 1, higher is better)")
        return {
            'method': 'Hierarchical_Agglomerative_Clustering',
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'discovered_clusters': self.discovered_clusters,
            'cluster_labels': self.cluster_labels,
            'quality_metrics': {
                'silhouette_score': silhouette if silhouette else 'N/A',
                'avg_cluster_size': float(np.mean([c['clause_count'] for c in self.discovered_clusters.values()]))
            }
        }
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _generate_cluster_name(self, top_terms: List[str]) -> str:
        # Legal risk theme detection
        themes = {
            'LIABILITY': ['liability', 'liable', 'damages', 'loss'],
            'INDEMNITY': ['indemnify', 'indemnification', 'hold', 'harmless'],
            'TERMINATION': ['terminate', 'termination', 'cancel', 'expire'],
            'IP': ['intellectual', 'property', 'patent', 'copyright'],
            'CONFIDENTIAL': ['confidential', 'nda', 'disclosure', 'secret'],
            'PAYMENT': ['payment', 'pay', 'fee', 'price'],
            'COMPLIANCE': ['comply', 'compliance', 'regulation', 'law'],
            'WARRANTY': ['warranty', 'warrant', 'represent', 'guarantee']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in term.lower() for term in top_terms[:5] for keyword in keywords):
                return f"RISK_{theme}"
        
        return f"RISK_{top_terms[0].upper()}_{top_terms[1].upper()}"


class DensityBasedRiskDiscovery:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, random_state: int = 42):
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        
        self.vectorizer = TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=3,
            max_df=0.85
        )
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='cosine',
            n_jobs=-1
        )
        
        self.discovered_clusters = {}
        self.cluster_labels = None
        self.feature_matrix = None
        self.outlier_indices = []
        
    def discover_risk_patterns(self, clauses: List[str], auto_tune: bool = True) -> Dict[str, Any]:
        print(f" Discovering risk patterns using DBSCAN...")
        
        cleaned_clauses = [self._clean_text(c) for c in clauses]
        print("   Creating TF-IDF feature matrix...")
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_clauses)
        feature_names = self.vectorizer.get_feature_names_out()
        if auto_tune:
            print("   Auto-tuning eps parameter...")
            self.eps = self._auto_tune_eps(self.feature_matrix)
            self.dbscan_model.eps = self.eps
            print(f"     Selected eps={self.eps:.3f}")
        print(f"   Fitting DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        self.cluster_labels = self.dbscan_model.fit_predict(self.feature_matrix)
        unique_clusters = [c for c in np.unique(self.cluster_labels) if c != -1]
        n_clusters = len(unique_clusters)
        n_noise = np.sum(self.cluster_labels == -1)
        print(f"   Found {n_clusters} clusters and {n_noise} outliers/noise points")
        print("   Analyzing discovered clusters...")
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_clauses = [clauses[i] for i in cluster_indices]
            cluster_tfidf = self.feature_matrix[cluster_mask].mean(axis=0)
            top_term_indices = np.argsort(np.asarray(cluster_tfidf).flatten())[-15:][::-1]
            top_terms = [feature_names[i] for i in top_term_indices]
            top_scores = [float(cluster_tfidf[0, i]) for i in top_term_indices]
            cluster_name = self._generate_cluster_name(top_terms, cluster_id)
            
            self.discovered_clusters[cluster_id] = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'top_terms': top_terms,
                'term_scores': top_scores,
                'clause_count': int(len(cluster_indices)),
                'proportion': float(len(cluster_indices) / len(clauses)),
                'is_core_cluster': len(cluster_indices) >= self.min_samples * 3
            }
        
        # Analyze outliers/noise
        self.outlier_indices = np.where(self.cluster_labels == -1)[0]
        outlier_clauses = [clauses[i] for i in self.outlier_indices]
        
        print(f" DBSCAN discovery complete: {n_clusters} clusters, {n_noise} outliers")
        
        return {
            'method': 'DBSCAN_Density_Based_Clustering',
            'n_clusters': n_clusters,
            'n_outliers': int(n_noise),
            'eps': self.eps,
            'min_samples': self.min_samples,
            'discovered_clusters': self.discovered_clusters,
            'cluster_labels': self.cluster_labels,
            'outlier_indices': self.outlier_indices.tolist(),
            'outlier_clauses': outlier_clauses[:10],  # First 10 outliers
            'quality_metrics': {
                'n_clusters': n_clusters,
                'outlier_ratio': float(n_noise / len(clauses)),
                'avg_cluster_size': float(np.mean([c['clause_count'] for c in self.discovered_clusters.values()])) if n_clusters > 0 else 0
            }
        }
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _auto_tune_eps(self, feature_matrix, sample_size: int = 1000) -> float:
        n_samples = min(sample_size, feature_matrix.shape[0])
        if feature_matrix.shape[0] > sample_size:
            indices = np.random.choice(feature_matrix.shape[0], sample_size, replace=False)
            sample_matrix = feature_matrix[indices]
        else:
            sample_matrix = feature_matrix
        
        k = self.min_samples
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(sample_matrix)
        distances, _ = nbrs.kneighbors(sample_matrix)
        
        k_distances = np.sort(distances[:, -1])
        
        eps = np.percentile(k_distances, 90)
        
        return float(eps)
    
    def _generate_cluster_name(self, top_terms: List[str], cluster_id: int) -> str:
        themes = {
            'LIABILITY': ['liability', 'liable', 'damages', 'loss'],
            'INDEMNITY': ['indemnify', 'indemnification', 'hold', 'harmless'],
            'TERMINATION': ['terminate', 'termination', 'cancel', 'expire'],
            'IP': ['intellectual', 'property', 'patent', 'copyright'],
            'CONFIDENTIAL': ['confidential', 'nda', 'disclosure', 'secret'],
            'PAYMENT': ['payment', 'pay', 'fee', 'price'],
            'COMPLIANCE': ['comply', 'compliance', 'regulation', 'law'],
            'WARRANTY': ['warranty', 'warrant', 'represent', 'guarantee']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in term.lower() for term in top_terms[:5] for keyword in keywords):
                return f"RISK_{theme}_C{cluster_id}"
        
        return f"RISK_CLUSTER_{cluster_id}_{top_terms[0].upper()}"
    
    def get_outlier_analysis(self) -> Dict[str, Any]:
        if len(self.outlier_indices) == 0:
            return {'message': 'No outliers found'}
        
        return {
            'n_outliers': len(self.outlier_indices),
            'outlier_ratio': len(self.outlier_indices) / len(self.cluster_labels),
            'interpretation': 'Outliers may represent rare or unique risk patterns that do not fit common categories'
        }


class NMFRiskDiscovery:
    def __init__(self, n_components: int = 7, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=3,
            max_df=0.85,
            norm='l2'  
        )
        import sklearn
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        
        nmf_params = {
            'n_components': n_components,
            'random_state': random_state,
            'init': 'nndsvda',
            'max_iter': 500
        }
        
        if sklearn_version >= (1, 0):
            # scikit-learn >= 1.0
            nmf_params['alpha_W'] = 0.1
            nmf_params['alpha_H'] = 0.1
            nmf_params['l1_ratio'] = 0.5
        elif sklearn_version >= (0, 19):
            nmf_params['alpha'] = 0.1
            nmf_params['l1_ratio'] = 0.5
        
        self.nmf_model = NMF(**nmf_params)
        
        self.discovered_components = {}
        self.component_labels = None
        self.feature_matrix = None
        self.W_matrix = None  
        self.H_matrix = None  
        
    def discover_risk_patterns(self, clauses: List[str]) -> Dict[str, Any]:
        print(f" Discovering risk patterns using NMF (n_components={self.n_components})...")
        cleaned_clauses = [self._clean_text(c) for c in clauses]
        print("   Creating TF-IDF feature matrix...")
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_clauses)
        feature_names = self.vectorizer.get_feature_names_out()
        print("   Fitting NMF model...")
        self.W_matrix = self.nmf_model.fit_transform(self.feature_matrix)
        self.H_matrix = self.nmf_model.components_
        
        self.component_labels = np.argmax(self.W_matrix, axis=1)
        
        print("   Extracting component keywords...")
        n_top_words = 15
        for component_idx in range(self.n_components):
            top_word_indices = np.argsort(self.H_matrix[component_idx])[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            top_weights = [self.H_matrix[component_idx][i] for i in top_word_indices]
            

            component_name = self._generate_component_name(top_words)
            

            clause_count = np.sum(self.component_labels == component_idx)
            

            avg_weight = np.mean(self.W_matrix[:, component_idx])
            
            self.discovered_components[component_idx] = {
                'component_id': component_idx,
                'component_name': component_name,
                'top_words': top_words,
                'word_weights': top_weights,
                'clause_count': int(clause_count),
                'proportion': float(clause_count / len(clauses)),
                'avg_strength': float(avg_weight)
            }
        

        reconstruction_error = self.nmf_model.reconstruction_err_
        

        sparsity = np.mean(self.W_matrix == 0)
        
        print(f" NMF discovery complete: {self.n_components} components found")
        print(f"   Reconstruction error: {reconstruction_error:.2f}")
        print(f"   Sparsity: {sparsity:.2%}")
        
        return {
            'method': 'NMF_Matrix_Factorization',
            'n_components': self.n_components,
            'discovered_components': self.discovered_components,
            'component_labels': self.component_labels,
            'component_strengths': self.W_matrix,
            'quality_metrics': {
                'reconstruction_error': float(reconstruction_error),
                'sparsity': float(sparsity),
                'avg_component_strength': float(np.mean(np.max(self.W_matrix, axis=1)))
            }
        }
    
    def get_clause_composition(self, clause_idx: int) -> Dict[int, float]:
        if self.W_matrix is None:
            return {}
        
        return {comp_id: float(weight) for comp_id, weight in enumerate(self.W_matrix[clause_idx])}
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _generate_component_name(self, top_words: List[str]) -> str:
        themes = {
            'LIABILITY': ['liability', 'liable', 'damages', 'loss'],
            'INDEMNITY': ['indemnify', 'indemnification', 'hold', 'harmless'],
            'TERMINATION': ['terminate', 'termination', 'cancel', 'expire'],
            'IP': ['intellectual', 'property', 'patent', 'copyright'],
            'CONFIDENTIAL': ['confidential', 'nda', 'disclosure', 'secret'],
            'PAYMENT': ['payment', 'pay', 'fee', 'price'],
            'COMPLIANCE': ['comply', 'compliance', 'regulation', 'law'],
            'WARRANTY': ['warranty', 'warrant', 'represent', 'guarantee']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in term.lower() for term in top_words[:5] for keyword in keywords):
                return f"COMPONENT_{theme}"
        
        return f"COMPONENT_{top_words[0].upper()}_{top_words[1].upper()}"


class SpectralClusteringRiskDiscovery:
    def __init__(self, n_clusters: int = 7, affinity: str = 'rbf', random_state: int = 42):
        self.n_clusters = n_clusters
        self.affinity = affinity  
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=3,
            max_df=0.85
        )

        from sklearn.cluster import SpectralClustering

        self.spectral_model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=random_state,
            n_init=10,
            assign_labels='kmeans'  # or 'discretize'
        )
        
        self.discovered_clusters = {}
        self.cluster_labels = None
        self.feature_matrix = None
        
    def discover_risk_patterns(self, clauses: List[str]) -> Dict[str, Any]:
        print(f" Discovering risk patterns using Spectral Clustering (n_clusters={self.n_clusters})...")
        

        cleaned_clauses = [self._clean_text(c) for c in clauses]
        

        print("   Creating TF-IDF feature matrix...")
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_clauses)
        feature_names = self.vectorizer.get_feature_names_out()
        

        print(f"   Fitting Spectral Clustering (affinity={self.affinity})...")
        print("     (This may take a while for large datasets...)")
        

        if self.feature_matrix.shape[0] > 5000:
            print(f"     Large dataset detected ({self.feature_matrix.shape[0]} clauses)")
            print("     Using nearest neighbors affinity for efficiency...")
            self.spectral_model.affinity = 'nearest_neighbors'
            self.spectral_model.n_neighbors = 10
        
        self.cluster_labels = self.spectral_model.fit_predict(self.feature_matrix)
        
        print("   Analyzing discovered clusters...")
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            cluster_clauses = [clauses[i] for i in cluster_indices]

            cluster_tfidf = self.feature_matrix[cluster_mask].mean(axis=0)
            top_term_indices = np.argsort(np.asarray(cluster_tfidf).flatten())[-15:][::-1]
            top_terms = [feature_names[i] for i in top_term_indices]
            top_scores = [float(cluster_tfidf[0, i]) for i in top_term_indices]

            cluster_name = self._generate_cluster_name(top_terms)
            
            self.discovered_clusters[cluster_id] = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'top_terms': top_terms,
                'term_scores': top_scores,
                'clause_count': int(len(cluster_indices)),
                'proportion': float(len(cluster_indices) / len(clauses))
            }

        if len(clauses) < 10000:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(self.feature_matrix, self.cluster_labels)
        else:
            silhouette = None
        
        print(f" Spectral clustering complete: {len(self.discovered_clusters)} clusters found")
        if silhouette:
            print(f"   Silhouette Score: {silhouette:.3f}")
        
        return {
            'method': 'Spectral_Clustering',
            'n_clusters': self.n_clusters,
            'affinity': self.affinity,
            'discovered_clusters': self.discovered_clusters,
            'cluster_labels': self.cluster_labels,
            'quality_metrics': {
                'silhouette_score': silhouette if silhouette else 'N/A',
                'n_clusters_found': len(self.discovered_clusters)
            }
        }
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _generate_cluster_name(self, top_terms: List[str]) -> str:
        themes = {
            'LIABILITY': ['liability', 'liable', 'damages', 'loss'],
            'INDEMNITY': ['indemnify', 'indemnification', 'hold', 'harmless'],
            'TERMINATION': ['terminate', 'termination', 'cancel', 'expire'],
            'IP': ['intellectual', 'property', 'patent', 'copyright'],
            'CONFIDENTIAL': ['confidential', 'nda', 'disclosure', 'secret'],
            'PAYMENT': ['payment', 'pay', 'fee', 'price'],
            'COMPLIANCE': ['comply', 'compliance', 'regulation', 'law'],
            'WARRANTY': ['warranty', 'warrant', 'represent', 'guarantee']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in term.lower() for term in top_terms[:5] for keyword in keywords):
                return f"SPECTRAL_{theme}"
        
        return f"SPECTRAL_{top_terms[0].upper()}_{top_terms[1].upper()}"


class GaussianMixtureRiskDiscovery:
    def __init__(self, n_components: int = 7, covariance_type: str = 'diag', random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type  
        self.random_state = random_state
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=3,
            max_df=0.85
        )
        
        from sklearn.mixture import GaussianMixture
        
        self.gmm_model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=10,
            max_iter=200
        )
        
        self.discovered_components = {}
        self.component_labels = None
        self.feature_matrix = None
        self.probabilities = None
        
    def discover_risk_patterns(self, clauses: List[str]) -> Dict[str, Any]:
        print(f" Discovering risk patterns using GMM (n_components={self.n_components})...")
        
        cleaned_clauses = [self._clean_text(c) for c in clauses]
        
        print("   Creating TF-IDF feature matrix...")
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_clauses)
        feature_names = self.vectorizer.get_feature_names_out()

        print("  Reducing dimensionality (GMM requires dense matrix)...")
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(100, self.feature_matrix.shape[1] - 1), random_state=self.random_state)
        X_reduced = svd.fit_transform(self.feature_matrix)

        print(f"   Fitting Gaussian Mixture Model (covariance={self.covariance_type})...")
        self.gmm_model.fit(X_reduced)

        self.component_labels = self.gmm_model.predict(X_reduced)
        self.probabilities = self.gmm_model.predict_proba(X_reduced)

        print("   Analyzing discovered components...")
        for component_id in range(self.n_components):
            component_mask = self.component_labels == component_id
            component_indices = np.where(component_mask)[0]
            
            if len(component_indices) == 0:
                continue

            component_clauses = [clauses[i] for i in component_indices]

            component_tfidf = self.feature_matrix[component_mask].mean(axis=0)
            top_term_indices = np.argsort(np.asarray(component_tfidf).flatten())[-15:][::-1]
            top_terms = [feature_names[i] for i in top_term_indices]
            top_scores = [float(component_tfidf[0, i]) for i in top_term_indices]

            component_name = self._generate_component_name(top_terms)

            avg_probability = np.mean(self.probabilities[component_mask, component_id])
            
            self.discovered_components[component_id] = {
                'component_id': component_id,
                'component_name': component_name,
                'top_terms': top_terms,
                'term_scores': top_scores,
                'clause_count': int(len(component_indices)),
                'proportion': float(len(component_indices) / len(clauses)),
                'avg_confidence': float(avg_probability)
            }

        bic = self.gmm_model.bic(X_reduced)
        aic = self.gmm_model.aic(X_reduced)
        
        print(f" GMM discovery complete: {len(self.discovered_components)} components found")
        print(f"   BIC: {bic:.2f} (lower is better)")
        print(f"   AIC: {aic:.2f} (lower is better)")
        
        return {
            'method': 'Gaussian_Mixture_Model',
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'discovered_components': self.discovered_components,
            'component_labels': self.component_labels,
            'probabilities': self.probabilities,
            'quality_metrics': {
                'bic': float(bic),
                'aic': float(aic),
                'avg_confidence': float(np.mean(np.max(self.probabilities, axis=1)))
            }
        }
    
    def get_clause_probabilities(self, clause_idx: int) -> Dict[int, float]:
        if self.probabilities is None:
            return {}
        
        return {comp_id: float(prob) for comp_id, prob in enumerate(self.probabilities[clause_idx])}
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _generate_component_name(self, top_terms: List[str]) -> str:
        themes = {
            'LIABILITY': ['liability', 'liable', 'damages', 'loss'],
            'INDEMNITY': ['indemnify', 'indemnification', 'hold', 'harmless'],
            'TERMINATION': ['terminate', 'termination', 'cancel', 'expire'],
            'IP': ['intellectual', 'property', 'patent', 'copyright'],
            'CONFIDENTIAL': ['confidential', 'nda', 'disclosure', 'secret'],
            'PAYMENT': ['payment', 'pay', 'fee', 'price'],
            'COMPLIANCE': ['comply', 'compliance', 'regulation', 'law'],
            'WARRANTY': ['warranty', 'warrant', 'represent', 'guarantee']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in term.lower() for term in top_terms[:5] for keyword in keywords):
                return f"GMM_{theme}"
        
        return f"GMM_{top_terms[0].upper()}_{top_terms[1].upper()}"


class MiniBatchKMeansRiskDiscovery:
    def __init__(self, n_clusters: int = 7, batch_size: int = 1000, random_state: int = 42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        
        from sklearn.cluster import MiniBatchKMeans

        self.kmeans_model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init=10,
            max_iter=300,
            reassignment_ratio=0.01
        )
        
        self.discovered_clusters = {}
        self.cluster_labels = None
        self.feature_matrix = None
        
    def discover_risk_patterns(self, clauses: List[str]) -> Dict[str, Any]:
        print(f" Discovering risk patterns using Mini-Batch K-Means (n_clusters={self.n_clusters})...")
        
        cleaned_clauses = [self._clean_text(c) for c in clauses]
        
        print("   Creating TF-IDF feature matrix...")
        self.feature_matrix = self.vectorizer.fit_transform(cleaned_clauses)
        feature_names = self.vectorizer.get_feature_names_out()

        print(f"   Fitting Mini-Batch K-Means (batch_size={self.batch_size})...")
        self.cluster_labels = self.kmeans_model.fit_predict(self.feature_matrix)

        print("   Analyzing discovered clusters...")
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue

            cluster_center = self.kmeans_model.cluster_centers_[cluster_id]

            top_term_indices = np.argsort(cluster_center)[-15:][::-1]
            top_terms = [feature_names[i] for i in top_term_indices]
            top_scores = [float(cluster_center[i]) for i in top_term_indices]

            cluster_name = self._generate_cluster_name(top_terms)

            from scipy.spatial.distance import cdist
            distances = cdist(
                self.feature_matrix[cluster_mask].toarray(),
                [cluster_center],
                metric='euclidean'
            )
            avg_distance = np.mean(distances)
            
            self.discovered_clusters[cluster_id] = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'top_terms': top_terms,
                'term_scores': top_scores,
                'clause_count': int(len(cluster_indices)),
                'proportion': float(len(cluster_indices) / len(clauses)),
                'avg_distance_to_center': float(avg_distance)
            }

        inertia = self.kmeans_model.inertia_
        
        print(f" Mini-Batch K-Means complete: {self.n_clusters} clusters found")
        print(f"   Inertia: {inertia:.2f} (lower is better)")
        print(f"   Speed boost vs standard K-Means: ~3-5x faster")
        
        return {
            'method': 'MiniBatch_KMeans',
            'n_clusters': self.n_clusters,
            'batch_size': self.batch_size,
            'discovered_clusters': self.discovered_clusters,
            'cluster_labels': self.cluster_labels,
            'quality_metrics': {
                'inertia': float(inertia),
                'avg_cluster_cohesion': float(np.mean([c['avg_distance_to_center'] for c in self.discovered_clusters.values()]))
            }
        }
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _generate_cluster_name(self, top_terms: List[str]) -> str:
        themes = {
            'LIABILITY': ['liability', 'liable', 'damages', 'loss'],
            'INDEMNITY': ['indemnify', 'indemnification', 'hold', 'harmless'],
            'TERMINATION': ['terminate', 'termination', 'cancel', 'expire'],
            'IP': ['intellectual', 'property', 'patent', 'copyright'],
            'CONFIDENTIAL': ['confidential', 'nda', 'disclosure', 'secret'],
            'PAYMENT': ['payment', 'pay', 'fee', 'price'],
            'COMPLIANCE': ['comply', 'compliance', 'regulation', 'law'],
            'WARRANTY': ['warranty', 'warrant', 'represent', 'guarantee']
        }
        
        for theme, keywords in themes.items():
            if any(keyword in term.lower() for term in top_terms[:5] for keyword in keywords):
                return f"MB_{theme}"
        
        return f"MB_{top_terms[0].upper()}_{top_terms[1].upper()}"

def compare_risk_discovery_methods(clauses: List[str], n_patterns: int = 7, 
                                   include_advanced: bool = True) -> Dict[str, Any]:
    print("="*80)
    print(" COMPARING RISK DISCOVERY METHODS")
    print(f"   Methods to test: {9 if include_advanced else 4}")
    print("="*80)
    
    results = {}
    
    print("\n" + "="*80)
    print("METHOD 1: K-Means Clustering (Original) - FAST")
    print("="*80)
    from risk_discovery import UnsupervisedRiskDiscovery
    kmeans_discovery = UnsupervisedRiskDiscovery(n_clusters=n_patterns)
    results['kmeans'] = kmeans_discovery.discover_risk_patterns(clauses)
    
    
    print("\n" + "="*80)
    print("METHOD 2: LDA Topic Modeling - PROBABILISTIC")
    print("="*80)
    lda_discovery = TopicModelingRiskDiscovery(n_topics=n_patterns)
    results['lda'] = lda_discovery.discover_risk_patterns(clauses)
    
    
    print("\n" + "="*80)
    print("METHOD 3: Hierarchical Clustering - STRUCTURE")
    print("="*80)
    hierarchical_discovery = HierarchicalRiskDiscovery(n_clusters=n_patterns)
    results['hierarchical'] = hierarchical_discovery.discover_risk_patterns(clauses)
    
    
    print("\n" + "="*80)
    print("METHOD 4: DBSCAN (Density-Based) - OUTLIERS")
    print("="*80)
    dbscan_discovery = DensityBasedRiskDiscovery(eps=0.3, min_samples=5)
    results['dbscan'] = dbscan_discovery.discover_risk_patterns(clauses, auto_tune=True)
    
    if include_advanced:
        
        print("\n" + "="*80)
        print("METHOD 5: NMF (Matrix Factorization) - PARTS-BASED")
        print("="*80)
        nmf_discovery = NMFRiskDiscovery(n_components=n_patterns)
        results['nmf'] = nmf_discovery.discover_risk_patterns(clauses)
        
        
        print("\n" + "="*80)
        print("METHOD 6: Spectral Clustering - GRAPH-BASED")
        print("="*80)
        spectral_discovery = SpectralClusteringRiskDiscovery(n_clusters=n_patterns)
        results['spectral'] = spectral_discovery.discover_risk_patterns(clauses)
        
        
        print("\n" + "="*80)
        print("METHOD 7: Gaussian Mixture Model - PROBABILISTIC SOFT")
        print("="*80)
        gmm_discovery = GaussianMixtureRiskDiscovery(n_components=n_patterns)
        results['gmm'] = gmm_discovery.discover_risk_patterns(clauses)
        
        
        print("\n" + "="*80)
        print("METHOD 8: Mini-Batch K-Means - ULTRA FAST")
        print("="*80)
        minibatch_discovery = MiniBatchKMeansRiskDiscovery(n_clusters=n_patterns)
        results['minibatch_kmeans'] = minibatch_discovery.discover_risk_patterns(clauses)
        
        
        print("\n" + "="*80)
        print("METHOD 9: Risk-o-meter (Doc2Vec + SVM) - PAPER BASELINE")
        print("="*80)
        print(" Based on: Chakrabarti et al., 2018")
        print("   Achievement: 91% accuracy on termination clauses")
        try:
            from risk_o_meter import RiskOMeterFramework
            risk_o_meter = RiskOMeterFramework(
                vector_size=100,
                epochs=30,
                verbose=True
            )
            results['risk_o_meter'] = risk_o_meter.discover_risk_patterns(clauses, n_patterns)
        except ImportError:
            print("  Risk-o-meter requires gensim. Install with: pip install gensim>=4.3.0")
            print("   Skipping Risk-o-meter comparison...")
        except Exception as e:
            print(f"  Risk-o-meter error: {e}")
            print("   Skipping Risk-o-meter comparison...")
    
    
    print("\n" + "="*80)
    print(" COMPARISON SUMMARY")
    print("="*80)
    
    summary = {
        'n_clauses': len(clauses),
        'target_patterns': n_patterns,
        'methods_compared': 9 if include_advanced else 4,
        'method_results': {}
    }
    
    for method_name, method_results in results.items():
        n_discovered = method_results.get('n_clusters') or method_results.get('n_topics', 0)
        
        print(f"\n{method_name.upper()}:")
        print(f"  Patterns Discovered: {n_discovered}")
        
        if 'quality_metrics' in method_results:
            print(f"  Quality Metrics: {method_results['quality_metrics']}")
        
        summary['method_results'][method_name] = {
            'n_patterns': n_discovered,
            'method': method_results['method'],
            'quality_metrics': method_results.get('quality_metrics', {})
        }
    
    print("\n" + "="*80)
    print(" COMPARISON COMPLETE")
    print("="*80)
    
    return {
        'summary': summary,
        'detailed_results': results
    }
