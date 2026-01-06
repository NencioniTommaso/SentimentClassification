import os
import pandas as pd
import string
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

# --- 1. Data Loading ---
def load_sentiment_data(base_path):
    data = []
    categories = {'pos': 1, 'neg': -1}
    for folder_name, sentiment_label in categories.items():
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path): continue

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        review_text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        review_text = f.read()
                data.append({'review': review_text, 'sentiment': sentiment_label})

    return pd.DataFrame(data)

# --- 2. Sentiment Experiment Class ---
class SentimentExperiment:
    def __init__(self, df):
        self.df = df
        self.results_log = []

    def _get_processed_tokens(self, text, use_negation=True, use_pos=False, 
                              filter_adjectives=False, use_position=False):
        """
        Tokenizes using NLTK and applies Negation, POS tags, and filtering.
        """
        tokens = nltk.word_tokenize(text)
        
        # 1. Handle Negation
        if use_negation:
            negation_words = {'not', 'no', 'never', "n't", 'hardly', 'scarcely', 'barely'}
            punctuation_set = set(string.punctuation)
            processed_tokens = []
            negation_scope = False
            for t in tokens:
                if t.lower() in negation_words:
                    negation_scope = True
                    processed_tokens.append(t)
                elif t in punctuation_set:
                    negation_scope = False
                    processed_tokens.append(t)
                else:
                    processed_tokens.append(f"NOT_{t}" if negation_scope else t)
            tokens = processed_tokens

        # 2. Part-of-Speech Tagging
        # Required for both Row (5) and Row (6)
        if use_pos or filter_adjectives:
            tagged = nltk.pos_tag(tokens)
            
            # Row (6): Use Adjectives only
            if filter_adjectives:
                # JJ, JJR, JJS are standard Penn Treebank tags for adjectives
                tokens = [word for word, tag in tagged if tag.startswith('JJ')]
            elif use_pos:
                # Row (5): Append POS tags to words
                tokens = [f"{word}_{tag}" for word, tag in tagged]

        # 3. Positional Tagging
        if use_position:
            n = len(tokens)
            pos_tokens = []
            for i, t in enumerate(tokens):
                if i < n // 4:
                    pos_tokens.append(f"{t}_first")
                elif i > (3 * n) // 4:
                    pos_tokens.append(f"{t}_last")
                else:
                    pos_tokens.append(f"{t}_mid")
            tokens = pos_tokens

        return " ".join(tokens)

    def run_configuration(self, label, ngram_range=(1, 1), use_presence=True, 
                          use_negation=True, use_pos=False, filter_adjectives=False,
                          use_position=False, max_features=None, min_df=4):
        """
        Reproduces all configurations from Figure 3.
        """
        # Pre-process text based on flags
        docs = self.df['review'].apply(
            lambda x: self._get_processed_tokens(x, use_negation, use_pos, 
                                                 filter_adjectives, use_position)
        )

        # Scikit-learn Vectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            binary=use_presence,
            min_df=min_df if max_features is None else 1, # Use parameter min_df unless max_features is set
            max_features=max_features,               # Required for Row (7) 
            token_pattern=r'\S+'
        )
        
        X = vectorizer.fit_transform(docs)
        y = self.df['sentiment'].values

        models = {
            'NB': MultinomialNB(),
            'Perceptron': Perceptron(max_iter=1000, random_state=42)
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        res = {"Setting": label, "Features": X.shape[1]}
        
        print(f"Executing: {label}...")
        for name, clf in models.items():
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            res[name] = round(np.mean(scores)*100, 2)
        
        self.results_log.append(res)
        return res
