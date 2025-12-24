import os
import pandas as pd
import string
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

    def _get_processed_tokens(self, text, use_negation=True, use_pos=False):
        """
        Tokenizes using NLTK and applies Negation and/or POS tags.
        """
        # Tokenize with NLTK (handles contractions like n't correctly)
        tokens = nltk.word_tokenize(text)
        
        # Handle Negation (Row 2 logic)
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

        # Handle POS Tagging (Row 5 logic)
        if use_pos:
            tagged = nltk.pos_tag(tokens)
            tokens = [f"{word}_{tag}" for word, tag in tagged]

        return " ".join(tokens)

    def run_configuration(self, label, ngram_range=(1, 1), use_presence=True, 
                          use_negation=True, use_pos=False):
        """
        Reproduces a specific configuration from Pang et al. (2002) Figure 3.
        """
        # Pre-process text based on flags
        docs = self.df['review'].apply(
            lambda x: self._get_processed_tokens(x, use_negation, use_pos)
        )

        # Scikit-learn Vectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            binary=use_presence,  # True = Presence, False = Frequency
            min_df=4,             # Feature cutoff used in paper
            token_pattern=r'\S+'  # Capture tokens precisely as processed above
        )
        
        X = vectorizer.fit_transform(docs)
        y = self.df['sentiment'].values

        # Classifiers for exam: Naive Bayes, Perceptron, Adaboost
        models = {
            'NB': MultinomialNB(),
            'Perceptron': Perceptron(max_iter=1000, random_state=42),
            'AdaBoost': AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1), 
                n_estimators=50, random_state=42
            )
        }

        # 3-fold cross-validation as per paper
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        res = {"Setting": label, "Features": X.shape[1]}
        
        print(f"Executing: {label}...")
        for name, clf in models.items():
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            res[name] = f"{np.mean(scores)*100:.2f}%"
        
        self.results_log.append(res)
        return res

# ==============================================================================
# --- Main Logic ---
# ==============================================================================

data_path = 'data/tokens'
df_reviews = load_sentiment_data(data_path)

if not df_reviews.empty:
    exp = SentimentExperiment(df_reviews)

    # Replicating Row (1) & (2): Presence vs Frequency
    exp.run_configuration("Unigrams Freq", use_presence=False)
    exp.run_configuration("Unigrams Pres", use_presence=True)
    
    # Replicating Row (3): Bigrams
    exp.run_configuration("Unigrams+Bigrams", ngram_range=(1, 2))
    
    # Replicating Row (5): POS Tagging
    exp.run_configuration("Unigrams+POS", use_pos=True)

    print("\n--- Comparison ---")
    print(pd.DataFrame(exp.results_log).to_string(index=False))