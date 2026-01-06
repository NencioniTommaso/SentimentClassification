import string
import nltk
from sentiment_module import SentimentExperiment

try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

def test_transformations():
    exp = SentimentExperiment(None)
    
    test_sentences = {
        "Negation": "This movie is not good and I didn't like it.",
        "POS Tagging": "I love this movie and the acting is great.",
        "Adjectives only": "The plot was boring but the cinematography was brilliant.",
        "Position": "Beginning of the movie. Middle part. End of the review."
    }

    print("--- PROCESSING TEST ---\n")

    # 1. Negation Test (Row 1-2 logic)
    text = test_sentences["Negazione"]
    processed = exp._get_processed_tokens(text, use_negation=True)
    print(f"ORIGINALE: {text}")
    print(f"NEGAZIONE: {processed}\n")

    # 2. POS Tagging Test (Row 5)
    text = test_sentences["POS Tagging"]
    processed = exp._get_processed_tokens(text, use_negation=False, use_pos=True)
    print(f"ORIGINALE: {text}")
    print(f"POS TAGS : {processed}\n")

    # 3. Adjectives Only test (Row 6)
    text = test_sentences["Soli Aggettivi"]
    processed = exp._get_processed_tokens(text, use_negation=False, filter_adjectives=True)
    print(f"ORIGINALE: {text}")
    print(f"AGGETTIVI: {processed}\n")

    # 4. Position Test (Row 8)
    text = test_sentences["Posizione"]
    processed = exp._get_processed_tokens(text, use_negation=False, use_position=True)
    print(f"ORIGINALE: {text}")
    print(f"POSIZIONE: {processed}\n")

if __name__ == "__main__":
    test_transformations()