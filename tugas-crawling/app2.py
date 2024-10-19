import streamlit as st
import pandas as pd
import pickle
import re

# Fungsi untuk cleansing text
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'http\S+|www\S+|https\S+', '', string, flags=re.MULTILINE)  # Hapus URL
    string = re.sub(r'@\w+', '', string)  # Hapus mention
    string = re.sub(r'#(\w+)', r'\1', string)  # Hapus simbol hashtag
    string = re.sub(r'[^a-zA-Z\s]', '', string)  # Hanya huruf
    string = re.sub(r'\s+', ' ', string).strip()  # Hapus spasi berlebih
    return string

# Fungsi untuk memprediksi sentimen
def predict_sentiment(text, model, feature_extractor):
    # cleansing text
    cleaned_text = cleansing(text)
    # extract feature using the loaded feature extractor (e.g., TF-IDF or BoW)
    text_feature = feature_extractor.transform([cleaned_text])
    # predict using the loaded model
    prediction = model.predict(text_feature)[0]
    return prediction

# Judul Aplikasi
st.title("NLP Dashboard - Predict Sentiment from Input Text")

# Membaca dataset CSV tanpa upload
csv_file = 'dataset_predicted_sentiment_new.csv'  # pastikan file ini sudah ada di directory yang benar
data = pd.read_csv(csv_file)
st.subheader("Dataset Sample")
st.write(data.head())  # Menampilkan beberapa sample dari dataset

# Load model dan fitur ekstraksi yang sudah disimpan
feature_bow = pickle.load(open('feature-bow.p', 'rb'))
model_nb = pickle.load(open('model-nb.p', 'rb'))

# Input teks dari user
st.subheader("Masukkan Teks untuk Analisis Sentimen")
user_input = st.text_area("Tulis teks di sini:")

# Tombol untuk analisis
if st.button('Analisis Sentimen'):
    if user_input:
        # Prediksi sentiment
        sentiment = predict_sentiment(user_input, model_nb, feature_bow)
        
        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi Sentimen:")
        st.write(f"Teks: {user_input}")
        st.write(f"Prediksi Sentimen: {sentiment}")
    else:
        st.error("Masukkan teks terlebih dahulu untuk analisis.")
