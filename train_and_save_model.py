import nltk
import numpy as np
import pickle
import os
from nltk.corpus import twitter_samples, stopwords
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

print("🔥 TRAINING PROPER MODEL WITH BETTER WEIGHTS...")

# Download data
nltk.download('twitter_samples', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # Clean text
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?://\S+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'[^\w\s]', ' ', tweet)
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and 
            word not in string.punctuation and
            len(word) > 2):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, Y, alpha=0.01, num_iters=5000, regularization=0.1):
    m, n = X.shape
    theta = np.zeros((n, 1))
    
    print("Training model...")
    for i in range(num_iters):
        z = np.dot(X, theta)
        h = sigmoid(z)
        
        # Compute cost with regularization
        cost = (-1/m) * np.sum(Y * np.log(h + 1e-8) + (1 - Y) * np.log(1 - h + 1e-8))
        cost += (regularization/(2*m)) * np.sum(theta[1:]**2)  # L2 regularization
        
        # Compute gradient with regularization
        gradient = (1/m) * np.dot(X.T, (h - Y))
        gradient[1:] += (regularization/m) * theta[1:]  # Don't regularize bias term
        
        # Update parameters
        theta = theta - alpha * gradient
        
        if i % 1000 == 0:
            print(f"  Iteration {i}: Cost = {cost:.6f}")
    
    return theta, cost

print("Loading and preparing data...")
all_positive = twitter_samples.strings('positive_tweets.json')
all_negative = twitter_samples.strings('negative_tweets.json')

# Use balanced data
train_pos = all_positive[:4000]
train_neg = all_negative[:4000]

train_x = train_pos + train_neg
train_y = np.append(np.ones((len(train_pos), 1)), 
                   np.zeros((len(train_neg), 1)), axis=0)

print(f"Training on {len(train_x)} tweets...")

print("Building frequency dictionary...")
freqs = build_freqs(train_x, train_y)
print(f"Created frequency dictionary with {len(freqs)} entries")

# Build feature matrix - Use LOG of frequencies to prevent large numbers
print("Building features...")
X = np.ones((len(train_x), 3))

for i, tweet in enumerate(train_x):
    words = process_tweet(tweet)
    pos_score = 0
    neg_score = 0
    
    for word in words:
        pos_score += freqs.get((word, 1.0), 0)
        neg_score += freqs.get((word, 0.0), 0)
    
    # Use log transformation to handle large numbers
    X[i, 1] = np.log(pos_score + 1)  # +1 to avoid log(0)
    X[i, 2] = np.log(neg_score + 1)

print("Training logistic regression with regularization...")
theta, final_cost = gradient_descent(X, train_y, alpha=0.1, num_iters=5000, regularization=0.1)

print(f"\n🎉 TRAINING COMPLETE!")
print(f"Final cost: {final_cost:.6f}")
print(f"Theta parameters: {theta.flatten()}")

# Test the trained model
def predict_text(text, freqs, theta):
    words = process_tweet(text)
    x = np.ones((1, 3))
    
    pos_score = 0
    neg_score = 0
    for word in words:
        pos_score += freqs.get((word, 1.0), 0)
        neg_score += freqs.get((word, 0.0), 0)
    
    # Use same transformation as training
    x[0, 1] = np.log(pos_score + 1)
    x[0, 2] = np.log(neg_score + 1)
    
    probability = sigmoid(np.dot(x, theta))[0][0]
    return probability

# Test with various examples
test_cases = [
    "I love this amazing product! It works perfectly!",
    "This is terrible and awful service! Worst experience ever!",
    "I hate this so much!",
    "Excellent quality and fantastic customer support!",
    "The package arrived on Tuesday",
    "Good product but expensive",
    "Waste of money, completely useless"
]

print("\n🧪 TESTING THE PROPER MODEL:")
for text in test_cases:
    prob = predict_text(text, freqs, theta)
    
    if prob > 0.7:
        sentiment = "POSITIVE"
    elif prob < 0.3:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    print(f"'{text}'")
    print(f"  → {sentiment} (probability: {prob:.4f})")
    print()

# Save the proper model
model_data = {
    'freqs': freqs,
    'theta': theta
}

os.makedirs('model', exist_ok=True)
with open('model/trained_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"💾 Proper model saved to 'model/trained_model.pkl'")
print(f"📊 Expected: Positive texts should have probability > 0.8")