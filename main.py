import pandas as pd
from sentiment_module import load_sentiment_data, SentimentExperiment

# ==============================================================================
# --- Main Logic ---
# ==============================================================================

if __name__ == "__main__":
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
        results_df.to_csv("sentiment_results.csv", index=False)
        print(results_df.to_string(index=False))
    else:
        print(f"Error: No data found in {data_path}. Check folder structure.")