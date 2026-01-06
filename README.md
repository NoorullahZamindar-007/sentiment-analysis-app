# sentiment-analysis-app
A web application for real-time sentiment analysis using Logistic Regression and NLP. Classifies text as Positive, Negative, or Neutral with confidence scores. Built with Python, Flask, and machine learning.
 
# Sentiment Analysis Web Application

A complete web application for real-time sentiment analysis using Logistic Regression and Natural Language Processing. This project classifies text as Positive, Negative, or Neutral with confidence scores. 

## Features    
  
- **Real-time Sentiment Analysis**: Instant classification of text input   v
- **Machine Learning Model** : Custom Logistic Regression trained on Twitter data       
- **Confidence Scoring** : Probability scores and confidence levels for predictions    
- **Web Interface**: Clean, responsive design built with Flask   
- **Example Testing**: Pre-loaded examples for quick testing  
- **Batch Processing**: Capability to analyze multiple texts at once    
   
## Technical Stack   

- **Backend**: Python, Flask 
- **Machine Learning**: Logistic Regression, NLTK  
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy
- **Model Persistence**: Pickle

## Model Performance

- **Training Data**: 8,000 Twitter samples (4,000 positive, 4,000 negative)
- **Accuracy**: >95% on test data
- **Features**: Word frequency analysis with text preprocessing
- **Algorithm**: Logistic Regression with gradient descent

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NoorullahZamindar-007/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (if needed)**
   ```bash
   python final_fix_proper.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and go to `http://localhost:5000`

### Using Pre-trained Model

The repository includes a pre-trained model. You can start the application directly without retraining.

## Project Structure

```
sentiment-analysis-app/
├── model/              # Trained model files
├── static/             # CSS and static files
├── templates/          # HTML templates
├── app.py              # Main Flask application
├── train_and_save_model.py    # Original training script
├── final_fix_proper.py        # Improved training script
├── debug_features.py          # Model debugging utility
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## API Endpoints

### Single Text Analysis
```http
POST /predict
Content-Type: application/json

{
    "text": "I love this amazing product!"
}
```

Response:
```json
{
    "text": "I love this amazing product!",
    "result": {
        "sentiment": "Positive",
        "confidence": 92.5,
        "probability": 0.925
    }
}
```

### Batch Analysis
```http
POST /batch_predict
Content-Type: application/json

{
    "texts": [
        "I love this!",
        "This is terrible",
        "It's okay"
    ]
}
```

### Health Check
```http
GET /health
```

## Model Details

### Training Process
1. **Data Collection**: Twitter samples from NLTK corpus
2. **Text Preprocessing**:
   - Remove URLs, mentions, hashtags
   - Tokenization and stemming
   - Stop words removal
3. **Feature Engineering**: Word frequency analysis
4. **Model Training**: Logistic Regression with gradient descent
5. **Model Evaluation**: Accuracy testing on holdout set

### Feature Extraction
- **Bias Term**: Constant feature
- **Positive Features**: Sum of positive word frequencies
- **Negative Features**: Sum of negative word frequencies
- **Normalization**: Log transformation for numerical stability

## Usage Examples

### Positive Sentiment
- "I love this amazing product!"
- "Excellent quality and fast shipping!"
- "Outstanding customer service!"

### Negative Sentiment  
- "This is terrible and awful service!"
- "Worst experience ever!"
- "Poor quality and bad support"

### Neutral Sentiment
- "The package arrived on Tuesday"
- "The product is as described"
- "It works as expected"

## Model Debugging

Use the debugging script to understand model predictions:
```bash
python debug_features.py
```

This shows detailed feature extraction and probability calculations for any text input.

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment, consider using:
- **WSGI Server**: Gunicorn or uWSGI
- **Reverse Proxy**: Nginx
- **Process Manager**: Systemd or Supervisor

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Performance Optimization

- Model loading optimized with pickle
- Feature extraction caching
- Efficient text preprocessing pipeline
- Minimal dependencies for fast deployment

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **NLTK Project** for the Twitter samples dataset
- **Flask Community** for the excellent web framework
- **NumPy** for numerical computing capabilities

## Support

If you have any questions or run into issues, please open an issue on GitHub.

## Future Enhancements

- Support for multiple languages
- Deep learning model integration
- Real-time streaming analysis
- Mobile application
- API rate limiting
- User authentication
- Historical analysis dashboard
