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
            'Perceptron': Perceptron(max_iter=1000, random_state=42),
            'AdaBoost': AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1), 
                n_estimators=50, random_state=42
            )
        }

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

    # (1) Unigrams Frequency: Count occurrences of words 
    exp.run_configuration("(1) Unigrams Freq", use_presence=False)

    # (2) Unigrams Presence: Binary indicator
    exp.run_configuration("(2) Unigrams Pres", use_presence=True)

    # (3) Unigrams + Bigrams: Combination of single words and word pairs
    exp.run_configuration("(3) Unigrams+Bigrams", ngram_range=(1, 2), use_negation=False)

    # (4) Bigrams only: Using only word pairs
    exp.run_configuration("(4) Bigrams only", ngram_range=(2, 2), use_negation=False, min_df=7)

    # (5) Unigrams + POS: Appending Part-of-Speech tags to words
    exp.run_configuration("(5) Unigrams+POS", use_pos=True)

    # (6) Adjectives only: Filtering text to keep only descriptive words
    exp.run_configuration("(6) Adjectives only", filter_adjectives=True)

    # (7) Top 2633 Unigrams: Most frequent unigrams
    exp.run_configuration("(7) Top 2633 Unigrams", max_features=2633)

    # (8) Unigrams + Position: Tagging words based on document quarter
    exp.run_configuration("(8) Unigrams+Position", use_position=True)

    # --- Summary Table ---
    print("\n--- Sentiment Classification Results ---")
    results_df = pd.DataFrame(exp.results_log)
    print(results_df.to_string(index=False))