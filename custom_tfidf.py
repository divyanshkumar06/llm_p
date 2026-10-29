import math
import re
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sp

class CustomTFIDFVectorizer:
    """
    Custom TF-IDF Vectorizer built strictly from scratch without sklearn.
    
    Features:
    - Word Unigrams & Word Bigrams
    - Character 2-grams, 3-grams, and 4-grams
    - Smooth IDF calculation: log((1 + N) / (1 + DF)) + 1
    - Sublinear TF scaling: 1 + log(tf) for tf > 0
    - L2 Row Normalization
    """
    def __init__(self, max_features=35000, sublinear_tf=True, norm='l2'):
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.idf_ = None
        self.n_samples_ = 0

    def _extract_ngrams(self, text):
        """
        Extracts word unigrams, word bigrams, and character 2,3,4-grams.
        """
        ngrams = []
        
        # Lowercase and clean words
        words = re.findall(r'\w+', text.lower(), flags=re.UNICODE)
        
        # 1. Word Unigrams
        for w in words:
            ngrams.append(f"w1_{w}")
            
        # 2. Word Bigrams
        for i in range(len(words) - 1):
            ngrams.append(f"w2_{words[i]}_{words[i+1]}")
            
        # 3. Character 2-grams, 3-grams, 4-grams
        clean_text = re.sub(r'\s+', ' ', text).strip()
        n = len(clean_text)
        
        # Char 2-grams
        for i in range(n - 1):
            ngrams.append(f"c2_{clean_text[i:i+2]}")
            
        # Char 3-grams
        for i in range(n - 2):
            ngrams.append(f"c3_{clean_text[i:i+3]}")
            
        # Char 4-grams
        for i in range(n - 3):
            ngrams.append(f"c4_{clean_text[i:i+4]}")
            
        return ngrams

    def fit(self, raw_documents):
        """
        Builds vocabulary and computes IDF vector from raw documents.
        """
        self.n_samples_ = len(raw_documents)
        df_counter = Counter()
        
        for doc in raw_documents:
            feature_set = set(self._extract_ngrams(doc))
            df_counter.update(feature_set)
            
        # Select top max_features by Document Frequency (DF)
        if self.max_features is not None and len(df_counter) > self.max_features:
            most_common = df_counter.most_common(self.max_features)
            selected_features = [feat for feat, count in most_common]
        else:
            selected_features = list(df_counter.keys())
            
        # Sort features for consistent ordering
        selected_features.sort()
        self.vocabulary_ = {feat: idx for idx, feat in enumerate(selected_features)}
        self.feature_names_ = selected_features
        
        # Compute smooth IDF: log((1 + N) / (1 + DF)) + 1
        n_features = len(selected_features)
        self.idf_ = np.zeros(n_features, dtype=np.float32)
        for feat, idx in self.vocabulary_.items():
            df = df_counter[feat]
            self.idf_[idx] = math.log((1.0 + self.n_samples_) / (1.0 + df)) + 1.0
            
        return self

    def transform(self, raw_documents):
        """
        Transforms raw documents to a TF-IDF sparse matrix (scipy CSR matrix).
        """
        if not self.vocabulary_ or self.idf_ is None:
            raise ValueError("The TF-IDF vectorizer has not been fitted yet!")
            
        n_docs = len(raw_documents)
        n_features = len(self.vocabulary_)
        
        indptr = [0]
        indices = []
        data = []
        
        vocab = self.vocabulary_
        
        for doc in raw_documents:
            ngrams = self._extract_ngrams(doc)
            term_counts = Counter(ngrams)
            
            row_indices = []
            row_data = []
            
            for token, count in term_counts.items():
                if token in vocab:
                    idx = vocab[token]
                    # Sublinear TF scaling: 1 + log(count)
                    if self.sublinear_tf:
                        tf = 1.0 + math.log(count)
                    else:
                        tf = float(count)
                        
                    tfidf_val = tf * self.idf_[idx]
                    row_indices.append(idx)
                    row_data.append(tfidf_val)
                    
            # Sort row entries by feature index
            if row_indices:
                sorted_pairs = sorted(zip(row_indices, row_data))
                row_indices = [p[0] for p in sorted_pairs]
                row_data = [p[1] for p in sorted_pairs]
                
                # Apply L2 Normalization if requested
                if self.norm == 'l2':
                    sq_sum = math.sqrt(sum(v * v for v in row_data))
                    if sq_sum > 0:
                        row_data = [v / sq_sum for v in row_data]
                        
            indices.extend(row_indices)
            data.extend(row_data)
            indptr.append(len(indices))
            
        X = sp.csr_matrix((data, indices, indptr), shape=(n_docs, n_features), dtype=np.float32)
        return X

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

if __name__ == "__main__":
    # Quick sanity check
    docs = ["नमस्ते दुनिया", "Hello world from India", "नमस्ते आप कैसे हैं"]
    vec = CustomTFIDFVectorizer(max_features=100)
    X = vec.fit_transform(docs)
    print("TF-IDF shape:", X.shape)
    print("Sample feature names:", vec.feature_names_[:5])
