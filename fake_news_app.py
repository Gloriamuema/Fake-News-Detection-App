# =========================================================
# 📰 Fake News Detection System with Custom UI
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================
# SETUP
# =========================================================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab', quiet=True)

ps = PorterStemmer()

def preprocess(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [
        ps.stem(word)
        for word in tokens
        if word.isalnum()
        and word not in stopwords.words('english')
        and word not in string.punctuation
    ]
    return " ".join(tokens)

# =========================================================
# MODEL TRAINING / LOADING
# =========================================================
@st.cache_resource
def train_or_load_model():
    if os.path.exists("fake_news_model.pkl") and os.path.exists("vectorizer.pkl"):
        model = pickle.load(open("fake_news_model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer

    st.info("🧠 Training model... This may take a few minutes.")

    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    # Label and balance
    fake["label"] = 0
    true["label"] = 1
    min_len = min(len(fake), len(true))
    fake = fake.sample(min_len, random_state=42)
    true = true.sample(min_len, random_state=42)

    df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)
    df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
    df = df[["content", "label"]].dropna()

    df["transformed"] = df["content"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["transformed"], df["label"], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Use balanced class weight
    model = LogisticRegression(max_iter=300, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, pred)
    st.success(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")

    pickle.dump(model, open("fake_news_model.pkl", "wb"))
    pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

    return model, tfidf


model, vectorizer = train_or_load_model()

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Fake News Detection", page_icon="🧠", layout="wide")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
    <style>
    body { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #065f46; text-align: center; font-weight: 700; }
    .stTextArea textarea { border-radius: 10px; border: 1px solid #d1d5db; font-size: 1rem; padding: 1rem; }
    .stButton>button { background-color: #065f46; color: white; border-radius: 10px; padding: 0.6rem 1.2rem; border: none; font-weight: 600; transition: 0.3s; }
    .stButton>button:hover { background-color: #047857; transform: scale(1.02); }
    .css-1d391kg { background-color: #ffffff !important; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("<h1>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.write("A machine learning-powered app that classifies news articles as **Real** or **Fake** using NLP.")

st.divider()

# =========================================================
# ABOUT
# =========================================================
with st.expander("ℹ️ About this App"):
    st.write("""
    This application uses **Logistic Regression** with **balanced training** and **TF-IDF vectorization**.
    Dataset: Kaggle Fake and Real News Dataset.
    """)

# =========================================================
# SINGLE PREDICTION
# =========================================================
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

with tab1:
    st.subheader("Single News Classification")
    news_input = st.text_area("✍️ Enter news text:", height=200)

    if st.button("Analyze"):
        if not news_input.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            transformed = preprocess(news_input)
            vectorized = vectorizer.transform([transformed])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized).max() * 100

            st.divider()
            if prediction == 1:
                st.success(f"✅ The news is **REAL** ({probability:.2f}% confidence).")
            else:
                st.error(f"🚫 The news is **FAKE** ({probability:.2f}% confidence).")

# =========================================================
# BATCH PREDICTION
# =========================================================
with tab2:
    st.subheader("Batch Classification")
    st.info("Upload a CSV file with a 'title' or 'text' column.")
    uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns and "title" not in df.columns:
            st.error("❌ File must contain a 'text' or 'title' column.")
        else:
            df["content"] = df.get("title", "") + " " + df.get("text", "")
            df["transformed"] = df["content"].apply(preprocess)
            vectors = vectorizer.transform(df["transformed"])

            df["prediction"] = model.predict(vectors)
            df["confidence"] = model.predict_proba(vectors).max(axis=1) * 100
            df["prediction_label"] = df["prediction"].apply(lambda x: "REAL" if x == 1 else "FAKE")

            st.dataframe(df[["content", "prediction_label", "confidence"]].head(10))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="predicted_fake_news.csv",
                mime="text/csv",
            )
