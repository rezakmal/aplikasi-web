import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

st.sidebar.markdown('Page 3: Dashboard :woman-raising-hand:')


# Fungsi untuk cleansing text
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'http\S+|www\S+|https\S+', '', string, flags=re.MULTILINE)  # Hapus URL
    string = re.sub(r'@\w+', '', string)  # Hapus mention
    string = re.sub(r'#(\w+)', r'\1', string)  # Hapus simbol hashtag
    string = re.sub(r'[^a-zA-Z\s]', '', string)  # Hanya huruf
    string = re.sub(r'\s+', ' ', string).strip()  # Hapus spasi berlebih
    return string

# Judul halaman
st.title('NLP Dashboard :bar_chart:')
st.write('Ini adalah data yang telah diolah, sehingga bisa menjadi lebih mudah dibaca dan dipahami.')

# Upload dataset
# uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")
file_path = 'data/dataset_predicted_sentiment.csv'
if file_path is not None:
    # Baca dataset
    data = pd.read_csv(file_path)

    # Rename kolom sesuai jika perlu
    if 'tweet' not in data.columns:
        data = data.rename(columns={'full_text': 'tweet'})
    
    # Cleansing data
    data['cleaned_tweet'] = data['tweet'].apply(cleansing)

    # Predict sentiment (dianggap ada kolom predicted_sentiment di CSV)
    if 'predicted_sentiment' in data.columns:
        sentiment_counts = data['predicted_sentiment'].value_counts()

        # Visualisasi Distribusi Sentimen
        st.subheader('Distribusi Sentimen')
        st.bar_chart(sentiment_counts)

        # WordCloud
        st.subheader('WordCloud')
        all_text = ' '.join(data['cleaned_tweet'].values)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Frekuensi Kata
        st.subheader('Frekuensi Kata')
        word_freq = pd.Series(' '.join(data['cleaned_tweet']).split()).value_counts()[:20]
        st.write(word_freq)

        # Tampilkan sampel data
        st.subheader('Sampel Data')
        st.dataframe(data[['tweet', 'cleaned_tweet', 'predicted_sentiment']].head(10))

    else:
        st.error("Dataset harus memiliki kolom 'predicted_sentiment'.")

else:
    st.warning("Silakan upload file CSV untuk analisis.")