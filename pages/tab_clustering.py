import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def tampilkan_tab_clustering(df):
    st.header("Clustering")
    st.info("Tab Clustering dikerjakan oleh M. Arrizal Rahman A.L.")

def render():
    st.header("Tab 3: Clustering Teks (K-Means)")
    st.write("Analisis pengelompokan postingan menggunakan representasi TF-IDF dan algoritma K-Means (3 cluster).")

    # 1. Memuat dataset bersih
    try:
        df = pd.read_csv('data/processed/dataset_timnas.csv')
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan di 'data/processed/dataset_timnas.csv'. Pastikan path sudah benar.")
        return

    # 2. Validasi kolom
    if 'text_clean' not in df.columns or 'text' not in df.columns:
        st.error("Kolom 'text_clean' atau 'text' tidak ditemukan dalam dataset.")
        return

    # 3. Representasi Teks TF-IDF dengan stopword khusus
    custom_stopwords = ['indonesia', 'inggris', 'timnas', 'yang', 'dan', 'di', 'ini', 'dari', 'ke', 'untuk', 'yang']
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=custom_stopwords)
    X_tfidf = vectorizer.fit_transform(df['text_clean'].fillna(''))

    # 4. Menerapkan Algoritma K-Means (3 Cluster)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_tfidf)

    # Mendapatkan kata dominan dari centroid
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    # 5. Menampilkan Hasil per Cluster
    for i in range(k):
        st.subheader(f"Cluster {i}")
        cluster_data = df[df['cluster'] == i]
        
        st.write(f"**Jumlah Postingan:** {len(cluster_data)} data")
        
        # 5 kata dominan
        top_terms = [terms[ind] for ind in order_centroids[i, :5]]
        st.write(f"**5 Kata Dominan:** {', '.join(top_terms)}")
        
        # 3 contoh postingan
        st.write("**Contoh Postingan:**")
        for sample in cluster_data['text'].head(3).values:
            st.markdown(f"- {sample}")
            
        st.markdown("---")

    # 6. Bagian Interpretasi Tema & Analisis Perbedaan Antar Cluster
    st.subheader("Interpretasi Tema & Analisis")
    st.markdown("""
    * **Tema Cluster 0:** *(Isi nama tema setelah melihat kata dominan di atas, misal: Apresiasi & Dukungan Suporter)*
    * **Tema Cluster 1:** *(Isi nama tema setelah melihat kata dominan di atas, misal: Kritik Taktik & Performa Tim)*
    * **Tema Cluster 2:** *(Isi nama tema setelah melihat kata dominan di atas, misal: Jadwal Pertandingan & Info Siaran)*
    
    **Perbedaan dan Tumpang Tindih:**
    * Setiap cluster mengelompokkan topik percakapan berdasarkan kemiripan kata kunci. 
    * Terdapat sedikit tumpang tindih pada kata-kata umum seputar sepak bola, namun pembagian tema utama sudah merefleksikan variasi opini dan informasi dari warganet.
    """)