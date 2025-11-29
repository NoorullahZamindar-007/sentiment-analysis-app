import pickle
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Load the model
with open('model/trained_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

freqs = model_data['freqs']
theta = model_data['theta']

print("🎯 Theta parameters:", theta.flatten())
print("📊 Total frequency entries:", len(freqs))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def debug_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # Clean text
    tweet_clean = re.sub(r'^RT[\s]+', '', tweet)
    tweet_clean = re.sub(r'https?://\S+', '', tweet_clean)
    tweet_clean = re.sub(r'#', '', tweet_clean)
    tweet_clean = re.sub(r'@\w+', '', tweet_clean)
    tweet_clean = re.sub(r'[^\w\s]', ' ', tweet_clean)
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet_clean)

    words_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and 
            word not in string.punctuation and
            len(word) > 2):
            stem_word = stemmer.stem(word)
            words_clean.append(stem_word)
    
    print(f"\n🔍 TEXT: '{tweet}'")
    print(f"📝 Processed words: {words_clean}")
    
    # Extract features
    x = np.ones((1, 3))
    pos_score = 0
    neg_score = 0
    
    print("\n📊 Word frequencies:")
    for word in words_clean:
        pos_freq = freqs.get((word, 1.0), 0)
        neg_freq = freqs.get((word, 0.0), 0)
        pos_score += pos_freq
        neg_score += neg_freq
        print(f"   '{word}': positive={pos_freq}, negative={neg_freq}")
    
    # Normalize
    x[0, 1] = pos_score / (len(words_clean) + 1) if words_clean else 0
    x[0, 2] = neg_score / (len(words_clean) + 1) if words_clean else 0
    
    print(f"\n🎯 Feature vector: {x}")
    print(f"   - Bias term: {x[0, 0]}")
    print(f"   - Positive score: {x[0, 1]:.4f}")
    print(f"   - Negative score: {x[0, 2]:.4f}")
    
    # Calculate prediction
    z = np.dot(x, theta)[0][0]
    probability = sigmoid(z)
    
    print(f"\n🧮 Calculation:")
    print(f"   z = {theta[0][0]:.6f}*{x[0, 0]:.1f} + {theta[1][0]:.6f}*{x[0, 1]:.4f} + {theta[2][0]:.6f}*{x[0, 2]:.4f}")
    print(f"   z = {z:.6f}")
    print(f"   Probability = sigmoid(z) = {probability:.4f}")
    
    # Determine sentiment
    if probability > 0.7:
        sentiment = "POSITIVE"
    elif probability < 0.3:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    print(f"\n🎯 FINAL RESULT: {sentiment} (probability: {probability:.4f})")
    return probability

# Test cases
test_texts = [
    "I love this amazing product! It works perfectly!",
    "This is terrible and awful service!",
    "I hate this so much!",
    "Excellent quality and fast shipping!"
]

for text in test_texts:
    debug_tweet(text)
    print("\n" + "="*50)