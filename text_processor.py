"""
text_processor.py
NLP pipeline for processing disaster text reports.
Uses TF-IDF vectorization for text feature extraction.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TextProcessor:
    def __init__(self, max_features=3000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.is_fitted = False

    def fit(self, texts):
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return self.tfidf_matrix

    def transform(self, texts):
        if not self.is_fitted:
            raise ValueError("TextProcessor not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return self.tfidf_matrix

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    @property
    def n_features(self):
        return self.tfidf_matrix.shape[1] if self.is_fitted else 0
