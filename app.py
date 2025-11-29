from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

app = Flask(__name__)

# Download NLTK data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Global variable to store our analyzer
analyzer = None

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.freqs = None
        self.theta = None
    
    def load_model(self, freqs, theta):
        self.freqs = freqs
        self.theta = theta
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def process_tweet(self, tweet):
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')
        
        # Better text cleaning
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
        tweet = re.sub(r'[^\w\s]', ' ', tweet)  # Replace punctuation with spaces
        
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and 
                word not in string.punctuation and
                len(word) > 2):  # Filter out short words
                stem_word = stemmer.stem(word)
                tweets_clean.append(stem_word)
        return tweets_clean
    
    def extract_features(self, tweet):
        if self.freqs is None:
            raise ValueError("Model not loaded")
            
        word_l = self.process_tweet(tweet)
        x = np.ones((1, 3))  # Start with bias term = 1
        
        pos_score = 0
        neg_score = 0
        for word in word_l:
            pos_score += self.freqs.get((word, 1.0), 0)
            neg_score += self.freqs.get((word, 0.0), 0)
        
        # Use log transformation like in the improved training
        x[0, 1] = np.log(pos_score + 1)  # +1 to avoid log(0)
        x[0, 2] = np.log(neg_score + 1)
        
        return x
    
    def predict_sentiment(self, text):
        if self.freqs is None or self.theta is None:
            return {
                'sentiment': 'Error',
                'confidence': 0,
                'probability': 0,
                'error': 'Model not loaded. Please train the model first.'
            }
            
        try:
            x = self.extract_features(text)
            probability = self.sigmoid(np.dot(x, self.theta))[0][0]
            
            # UPDATED: Better thresholds for sentiment classification
            if probability > 0.6:
                sentiment = "Positive"
                confidence = probability
            elif probability < 0.4:
                sentiment = "Negative" 
                confidence = 1 - probability
            else:
                sentiment = "Neutral"
                confidence = 1 - abs(probability - 0.5) * 2
                
            return {
                'sentiment': sentiment,
                'confidence': round(confidence * 100, 2),
                'probability': round(probability, 4)
            }
        except Exception as e:
            return {
                'sentiment': 'Error',
                'confidence': 0,
                'probability': 0,
                'error': str(e)
            }

def load_model():
    global analyzer
    try:
        model_path = 'model/trained_model.pkl'
        
        if not os.path.exists(model_path):
            return {
                'success': False,
                'message': f"Model file not found at {model_path}. Please run the training script first."
            }
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if model data has the required keys
        if 'freqs' not in model_data or 'theta' not in model_data:
            return {
                'success': False,
                'message': "Model file is corrupted. Please retrain the model."
            }
        
        analyzer = SimpleSentimentAnalyzer()
        analyzer.load_model(model_data['freqs'], model_data['theta'])
        
        print("Model loaded successfully!")
        print(f"Frequency dictionary size: {len(model_data['freqs'])}")
        print(f"Theta parameters: {model_data['theta'].flatten()}")
        
        # Test the model with sample text
        test_text = "I love this amazing product!"
        test_result = analyzer.predict_sentiment(test_text)
        print(f"Test prediction: '{test_text}' -> {test_result['sentiment']} (prob: {test_result['probability']})")
        
        return {
            'success': True,
            'message': 'Model loaded successfully'
        }
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(error_msg)
        return {
            'success': False,
            'message': error_msg
        }

# Load model when app starts
print("Starting Flask app...")
model_status = load_model()
if model_status['success']:
    print("Model loaded successfully!")
else:
    print(f"{model_status['message']}")

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { 
                color: #333; 
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            textarea { 
                width: 100%; 
                height: 120px; 
                margin: 15px 0; 
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
                resize: vertical;
                font-family: Arial;
                box-sizing: border-box;
            }
            textarea:focus {
                outline: none;
                border-color: #007bff;
            }
            button { 
                padding: 12px 30px; 
                background: #007bff;
                color: white; 
                border: none; 
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                display: block;
                margin: 20px auto;
                transition: background-color 0.2s;
            }
            button:hover {
                background: #0056b3;
            }
            .result { 
                margin: 25px 0; 
                padding: 20px; 
                border-radius: 5px;
                border-left: 5px solid;
            }
            .Positive { 
                background: #d4edda; 
                border-color: #28a745;
                color: #155724;
            }
            .Negative { 
                background: #f8d7da; 
                border-color: #dc3545;
                color: #721c24;
            }
            .Neutral { 
                background: #fff3cd; 
                border-color: #ffc107;
                color: #856404;
            }
            .Error { 
                background: #f8d7da; 
                border-color: #dc3545;
                color: #721c24;
            }
            .examples {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 5px;
            }
            .examples h3 {
                margin-top: 0;
                color: #333;
                margin-bottom: 15px;
            }
            .example-btn {
                background: #6c757d;
                color: white;
                border: none;
                padding: 8px 15px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.2s;
            }
            .example-btn:hover {
                background: #545b62;
            }
            .hidden {
                display: none;
            }
            .loading {
                background: #e9ecef;
                border-color: #6c757d;
                color: #495057;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sentiment Analysis</h1>
            <p class="subtitle">Analyze the sentiment of your text using machine learning</p>
            
            <div>
                <label for="textInput"><strong>Enter text to analyze:</strong></label>
                <textarea id="textInput" placeholder="Type your text here... Example: 'I love this amazing product!' or 'This is terrible service'"></textarea>
            </div>
            
            <button onclick="analyzeSentiment()">Analyze Sentiment</button>
            
            <div id="result" class="hidden"></div>

            <div class="examples">
                <h3>Try these examples (click to test):</h3>
                <button class="example-btn" onclick="setExample('I love this amazing product! It works perfectly!')">Positive Example</button>
                <button class="example-btn" onclick="setExample('This is terrible and awful service! Worst experience ever!')">Negative Example</button>
                <button class="example-btn" onclick="setExample('The package arrived on Tuesday as expected.')">Neutral Example</button>
                <button class="example-btn" onclick="setExample('Excellent quality and fantastic customer support!')">Excellent Review</button>
                <button class="example-btn" onclick="setExample('Horrible experience, never buying again!')">Terrible Review</button>
                <button class="example-btn" onclick="setExample('Good product but delivery was late')">Mixed Feelings</button>
            </div>
        </div>
        
        <script>
            function setExample(text) {
                document.getElementById('textInput').value = text;
                document.getElementById('result').classList.add('hidden');
            }

            function analyzeSentiment() {
                const text = document.getElementById('textInput').value.trim();
                const resultDiv = document.getElementById('result');
                
                if (!text) {
                    alert('Please enter some text to analyze!');
                    return;
                }
                
                resultDiv.innerHTML = '<div class="result loading">Analyzing your text... Please wait.</div>';
                resultDiv.classList.remove('hidden');
                
                fetch('/predict', {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({text: text})
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok: ' + response.status);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        resultDiv.innerHTML = `<div class="result Error">Error: ${data.error}</div>`;
                    } else {
                        const result = data.result;
                        
                        resultDiv.innerHTML = `
                            <div class="result ${result.sentiment}">
                                <h3>Sentiment: ${result.sentiment}</h3>
                                <p><strong>Confidence:</strong> ${result.confidence}%</p>
                                <p><strong>Probability Score:</strong> ${result.probability}</p>
                                <p><strong>Text Analyzed:</strong> "${data.text}"</p>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    resultDiv.innerHTML = `<div class="result Error">Error: ${error.message}</div>`;
                });
            }

            // Allow Enter key to submit (with Shift+Enter for new line)
            document.getElementById('textInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    analyzeSentiment();
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
            
        if analyzer is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
            
        result = analyzer.predict_sentiment(text)
        
        return jsonify({
            'text': text,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    if analyzer is None:
        return jsonify({'status': 'Model not loaded', 'model_loaded': False})
    else:
        return jsonify({'status': 'Healthy', 'model_loaded': True})

@app.route('/model-info')
def model_info():
    if analyzer is not None and analyzer.freqs is not None:
        return jsonify({
            'model_loaded': True,
            'freqs_size': len(analyzer.freqs),
            'theta': analyzer.theta.flatten().tolist()
        })
    else:
        return jsonify({'model_loaded': False})

if __name__ == '__main__':
    print("Server starting on http://localhost:5000")
    print("Enter text on the webpage to analyze sentiment")
    app.run(debug=True, host='0.0.0.0', port=5000)