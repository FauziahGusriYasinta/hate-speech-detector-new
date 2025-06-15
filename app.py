import streamlit as st
import joblib
import numpy as np
import re, string
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Stopwords
stop_words = set(stopwords.words('indonesian'))

# Kamus normalisasi sederhana
normalization_dict = {
    'ga': 'tidak', 'gak': 'tidak', 'tdk': 'tidak', 'klo': 'kalau',
    'aja': 'saja', 'sy': 'saya', 'yg': 'yang', 'ya': 'iya', 'lo': 'kamu',
    'gw': 'aku', 'si': 'dia', 'jd': 'jadi', 'jgn': 'jangan',
    'tai': 'taik', 'taik': 'taik', 'bangsat': 'bangsat', 'goblok': 'goblok',
    'utk': 'untuk', 'bgt': 'banget', 'tp': 'tapi', 'trs': 'terus',
    'dgn': 'dengan', 'sm': 'sama', 'jg': 'juga', 'dr': 'dari', 'krn': 'karena',
}

def normalize_text(text):
    return ' '.join([normalization_dict.get(word, word) for word in text.split()])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?:\/\/\S+|@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = normalize_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Load model dan vectorizer
clf = joblib.load('modell_final_lr.pkl')
tfidf = joblib.load('tfidff_final.pkl')
bert = SentenceTransformer('distiluse-base-multilingual-cased-v2')  # ✅ model ringan & cocok untuk deploy!

# UI
st.title("🇮🇩 Deteksi Hate Speech Bahasa Indonesia")
st.write("Masukkan teks atau tweet di bawah ini:")

text_input = st.text_area("Input teks")

if st.button("Deteksi"):
    if not text_input.strip():
        st.warning("Teks tidak boleh kosong ya, sayang 🤍")
    else:
        cleaned = preprocess(text_input)
        tfidf_vec = tfidf.transform([cleaned])
        bert_vec = bert.encode([cleaned])
        fusion = np.hstack((tfidf_vec.toarray(), bert_vec))
        pred = clf.predict(fusion)[0]
        label = "💔 Hate Speech" if pred == 1 else "💖 Non-Hate Speech"
        st.success(f"Hasil Deteksi: {label}")
