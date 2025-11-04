# Fake-News-App
## 📰 Fake News Detection System

Table of Contents
1. Project Overview
2. Features
3. Technology Stack
4. Installation
5. Usage
6. Live Demo
7. Model Details
8. Dataset
9. Results
10. Screenshots
11. Contributing
12. License
## Project Overview
The Fake News Detection System is an AI-powered project that identifies whether a news article is real or fake using natural language processing (NLP) and machine learning.
It helps combat misinformation by allowing users to verify news articles quickly and efficiently.

## Features
1. Classify news articles as Real or Fake
2. Preprocess text: remove punctuation, stopwords and special characters
3. Feature extraction using TF-IDF Vectorizer
4. Supports multiple classifiers: Logistic Regression, Random Forest and others.
5. Evaluates performance using Accuracy, Precision, Recall, and F1-Score
6. Real-time predictions via a Streamlit web interface

## Technology Stack
1. Language: Python 3.x
2. Libraries: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn
3. Frontend: Streamlit for web interface
4. Version Control: Git & GitHub

## Installation
Clone the repository:
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection 

## Create a virtual environment:
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

## Install dependencies:
pip install -r requirements.txt

## Usage
1. Preprocess the dataset:
python preprocess.py

2. Train the machine learning model:
python train_model.py

3. Save the trained model and vectorizer as pickle files:
import pickle
pickle.dump(trained_model, open("model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("vectorizer.pkl", "wb"))


4. Run predictions directly from Python:
python predict.py --news "Your news article text here"

## Live Demo

You can run a Streamlit web app for real-time predictions:
1. Launch the app:
streamlit run app.py
2. Enter a news article headline or text in the input box.
3. Click Predict to see whether the news is Real or Fake.
You can deploy the Streamlit app online using Streamlit Cloud
 or Heroku for public access.

## Model Details

1. Algorithm: Logistic Regression (default)
2. Features: TF-IDF vectorized text
3. Evaluation Metrics:
Accuracy: ~95% (example)
Precision, Recall, F1-Score
4. Other classifiers like Random Forest, Naive Bayes, or SVM can also be added for comparison.

# Dataset
Public datasets of labeled real and fake news articles.
Example: Kaggle Fake News Dataset

## Columns typically include:
1. title – News headline
2. text – Full article content
3. label – 0 for Real, 1 for Fake

## Results
1. High accuracy in detecting fake news
2. Confusion matrix and classification reports help visualize model performance
3. Optional: SHAP or LIME explanations for model predictions

# Screenshots
Streamlit Web Interface

# Prediction Example

I will add screenshots in  form of screenshots or folder for better presentation.

# Contributing
Contributions are welcome! You can:
1. Add new ML models or classifiers
2. Improve preprocessing and feature extraction
3. Enhance the Streamlit interface for real-time predictions

Please fork the repo and submit a pull request.

Author 
Gloria Muema