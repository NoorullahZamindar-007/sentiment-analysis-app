import pickle
import numpy as np

def verify_model():
    try:
        with open('model/trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        print(" Model loaded successfully!")
        print(f" Frequency dictionary entries: {len(model_data['freqs'])}")
        print(f" Theta parameters: {model_data['theta'].flatten()}")
        
        # Show some sample frequencies
        print("\n Sample frequency entries:")
        sample_items = list(model_data['freqs'].items())[:5]
        for (word, sentiment), freq in sample_items:
            sentiment_str = "Positive" if sentiment == 1.0 else "Negative"
            print(f"   '{word}' ({sentiment_str}): {freq}")
            
    except FileNotFoundError:
        print(" Model file not found. Please run train_and_save_model.py first.")
    except Exception as e:
        print(f" Error loading model: {e}")

if __name__ == "__main__":
    verify_model()