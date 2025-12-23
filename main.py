'''
Useful links:
https://www.cs.cornell.edu/home/llee/papers/sentiment.home.html

https://www.cs.cornell.edu/people/pabo/movie-review-data/
The best idea is to use polarity dataset v.1.1 extremly similar to the original dataset
(in README.1.1 there is an interesting paragraph on using the stars (ratings) themselves to classify the reviw)
'''

import os
import pandas as pd
import string
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# --- 1. Data Loading Function ---

def load_sentiment_data(base_path):
    """
    Loads text reviews from 'pos' and 'neg' subdirectories into a DataFrame.
    Includes error handling for common file encoding issues.
    """
    data = []
    categories = {'pos': 1, 'neg': -1}
    
    for folder_name, sentiment_label in categories.items():
        folder_path = os.path.join(base_path, folder_name)
        print(f"Loading files from: {folder_path}")
        
        if not os.path.isdir(folder_path):
            print(f"Warning: Directory '{folder_path}' not found. Skipping.")
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                
                try:
                    # 1. Try reading with the standard 'utf-8' encoding
                    with open(file_path, 'r', encoding='utf-8') as f:
                        review_text = f.read()
                
                except UnicodeDecodeError:
                    # 2. If utf-8 fails, try 'latin-1' (often works for older English text)
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            review_text = f.read()
                        # print(f"Successfully loaded {file_name} using latin-1.") # Optional debug
                    except Exception as e:
                        # 3. If everything fails, log the error and skip the file
                        print(f"FATAL: Could not read file {file_name} with either utf-8 or latin-1: {e}")
                        continue
                
                data.append({
                    'review': review_text, 
                    'sentiment': sentiment_label
                })

    df = pd.DataFrame(data)
    
    print("\nData loaded successfully.")
    if not df.empty:
        print("Sentiment distribution:")
        print(df['sentiment'].value_counts())
    
    return df

# --- 2. Negation Tagging (Preprocessing) Function ---

def apply_negation_tagging(text):
    """
    Applies negation tagging as per Pang et al. (2002).
    Assumes punctuation is already space-separated.
    """
    negation_words = ['not', 'no', 'never', 'n\'t', 'hardly', 'scarcely', 'barely']
    punctuation_set = set(string.punctuation)
    tokens = [t for t in text.split(' ') if t] 
    
    new_tokens = []
    negation_scope = False
    
    for token in tokens:
        token_lower = token.lower()
        
        # Start negation scope
        if token_lower in negation_words:
            negation_scope = True
            new_tokens.append(token)
            continue
            
        # End negation scope at first punctuation mark
        if token in punctuation_set:
            negation_scope = False
            new_tokens.append(token)
            continue
            
        # Apply tag if in scope
        if negation_scope:
            new_tokens.append(f'NOT_{token}')
        else:
            new_tokens.append(token)
            
    return ' '.join(new_tokens)

# --- 3. Vectorization Function (Feature Extraction) ---

def vectorize_data(df, min_df_cutoff=4):
    """
    Converts text reviews into a sparse feature matrix using Unigram Presence.
    (Matches Pang et al. NB (2) result: 81.0%)
    """
    
    # token_pattern captures standard words and NOT_ tagged words
    vectorizer = CountVectorizer(
        binary=True, # Feature PRESENCE (1 or 0), not frequency
        min_df=min_df_cutoff, # Only tokens appearing in >= 4 documents
        token_pattern=r'\b\w+\b|\bNOT_\w+\b' 
    )
    
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment'].values
    
    print(f"\nFeature Matrix created: {X.shape[0]} documents, {X.shape[1]} features.")
    print(f"Vectorization method: Unigram Presence (binary=True, min_df={min_df_cutoff})")
    
    return X, y, vectorizer

# --- 4. Model Training and Evaluation Function ---

def evaluate_models(X, y, classifiers, cv):
    """
    Evaluates classifiers using cross-validation.
    ***THIS IS WHERE THE MODELS ARE TRAINED (via cross_val_score)***
    """
    results = {}
    print(f"\n--- Running {cv.n_splits}-Fold Cross-Validation ---")
    
    for name, model in classifiers.items():
        # cross_val_score handles the splitting, training (model.fit), and scoring 
        # for each of the 3 folds.
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        mean_accuracy_percent = np.mean(scores) * 100
        
        results[name] = {
            'scores': scores * 100,
            'mean_accuracy': mean_accuracy_percent
        }
        
        # Report accuracy as required by the paper (in percent)
        print(f"| {name:<20} | Mean Accuracy: {mean_accuracy_percent:.2f}% |")
        
    return results

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

# **CRITICAL FIX**: Use the corrected path you provided:
data_directory = 'data/tokens'

# 1. Load Data
df_reviews = load_sentiment_data(data_directory)

# Ensure data was loaded before proceeding
if df_reviews.empty:
    print("\nERROR: No data loaded. Please check your 'data/tokens/pos' and 'data/tokens/neg' directories.")
else:
    # 2. Apply Negation Tagging
    print("Applying negation tagging...")
    df_reviews['review'] = df_reviews['review'].apply(apply_negation_tagging)
    print("Negation tagging complete.")

    # 3. Feature Extraction (Vectorization)
    X_features, y_labels, feature_model = vectorize_data(df_reviews, min_df_cutoff=4)

    # 4. Define Classifiers and Cross-Validation Strategy
    weak_learner = DecisionTreeClassifier(max_depth=1)

    classifiers = {
        # Using the event model (CountVectorizer with binary=True) for Naive Bayes
        'Naive Bayes (NB)': MultinomialNB(), 
        # Using Perceptron as the simple linear classifier
        'Perceptron': Perceptron(max_iter=1000, tol=1e-3, random_state=42),
        'AdaBoost': AdaBoostClassifier(
            estimator=weak_learner, 
            n_estimators=50, # Standard number of boosting stages
            random_state=42
        )
    }
    
    # 5-fold cross-validation, maintaining class balance (as used in the paper)
    n_splits = 3
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 6. Train and Evaluate Models
    # This function call performs the training (fitting)
    results = evaluate_models(X_features, y_labels, classifiers, cv)