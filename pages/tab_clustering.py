import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


def tampilkan_tab_clustering(data):
    st.header("Clustering TF-IDF + K-Means")

    text_col = 'text_clean' if 'text_clean' in data.columns else 'text'

    df = data.dropna(subset=[text_col]).copy()
    df = df[df[text_col].astype(str).str.strip() != '']
    df = df.reset_index(drop=True)

    stopword_id = set("""
    garuda yang dan di ke dari untuk pada dengan adalah ini itu akan atau juga karena
    sebagai oleh dalam telah sudah saya kamu dia kami kita mereka anda nya ku mu
    tidak bukan belum jangan ada tak bisa dapat harus saja lebih sangat agar
    oleh terhadap antara setelah sebelum saat ketika jika maka bahwa para
    """.split())

    stopword_en = set("""
    the a an and or of to in on for with is are was were be been being this that
    these those it its as at by from have has had do does did but if then so
    """.split())

    slang_dict = {
        'yg':'yang','dgn':'dengan','utk':'untuk','tdk':'tidak','gak':'tidak',
        'ga':'tidak','gk':'tidak','bgt':'banget','aja':'saja','kalo':'kalau',
        'klo':'kalau','udh':'sudah','udah':'sudah','tp':'tapi','tpi':'tapi',
        'gw':'saya','gue':'saya','jg':'juga','dr':'dari'
    }

    def normalize_slang(t):
        return ' '.join([slang_dict.get(w, w) for w in t.split()])

    def remove_stopwords(t):
        return ' '.join([w for w in t.split()
                         if w not in stopword_id and w not in stopword_en and len(w) > 2])

    df['text_final'] = (df[text_col].astype(str)
                        .str.lower()
                        .apply(normalize_slang)
                        .apply(remove_stopwords))
    df = df[df['text_final'].str.strip() != ''].reset_index(drop=True)

    if len(df) == 0:
        st.warning("Tidak ada data teks yang bisa diproses setelah preprocessing.")
        return

    vectorizer = TfidfVectorizer(
        stop_words=list(stopword_id | stopword_en),
        max_features=1000, ngram_range=(1, 2), min_df=2
    )
    X = vectorizer.fit_transform(df['text_final'])

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    nama_tema = {
        0: "Kritik terhadap PSSI dan Kebijakan Pemain",
        1: "Diskusi Umum Seputar Timnas Indonesia",
        2: "#TimnasDay dan #PertiwiBerani",
    }

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    top_words = {i: [terms[j] for j in order_centroids[i, :5]] for i in range(k)}

    st.caption(f"Data valid: **{len(df)}** baris | Jumlah cluster: **{k}**")

    st.subheader("Ringkasan per Cluster")
    for c in range(k):
        subset = df[df['cluster'] == c]
        contoh = subset[text_col].head(3).tolist()

        with st.expander(f"Cluster {c} - {nama_tema[c]} ({len(subset)} postingan)"):
            st.write(f"**Kata dominan (TF-IDF)**: {', '.join(top_words[c])}")
            st.write("**3 contoh postingan:**")
            for i, t in enumerate(contoh, 1):
                st.write(f"**{i}.** {str(t)[:300]}")

    st.divider()

    with st.expander("Visualisasi K-Means Clustering (PCA 2D)", expanded=False):
        pca = PCA(n_components=2, random_state=0)
        reduced = pca.fit_transform(X.toarray())
        centers = pca.transform(kmeans.cluster_centers_)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(reduced[:, 0], reduced[:, 1], c=df['cluster'],
                   cmap='Set2', s=25, alpha=0.7)
        ax.scatter(centers[:, 0], centers[:, 1], marker='x',
                   s=150, c='black', label='Centroid')
        ax.set_title('Sebaran Cluster (PCA 2D)', fontsize=10)
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Word Cloud per Cluster", expanded=False):
        fig, axes = plt.subplots(1, k, figsize=(12, 3.5))
        for i in range(k):
            text_cluster = ' '.join(df[df['cluster'] == i]['text_final'].tolist())
            if text_cluster.strip():
                wc = WordCloud(width=500, height=300, background_color='white',
                               colormap='viridis', max_words=80).generate(text_cluster)
                axes[i].imshow(wc, interpolation='bilinear')
            axes[i].axis('off')
            axes[i].set_title(f'Cluster {i}', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Kata yang Sering Muncul per Cluster", expanded=False):
        fig, axes = plt.subplots(1, k, figsize=(12, 3.5))
        for i in range(k):
            all_words = ' '.join(df[df['cluster'] == i]['text_final'].tolist()).split()
            common = Counter(all_words).most_common(10)
            if common:
                words, counts = zip(*common)
                axes[i].barh(list(words)[::-1], list(counts)[::-1], color='steelblue')
                axes[i].set_title(f'Cluster {i} - Top 10 Kata', fontsize=9)
                axes[i].set_xlabel('Frekuensi', fontsize=8)
                axes[i].tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    st.subheader("Penjelasan Perbedaan dan Tumpang Tindih")

    col_kiri, col_kanan = st.columns(2)

    with col_kiri:
        with st.expander("Perbedaan antar Cluster", expanded=False):
            st.markdown("""
**Cluster 0 - Kritik terhadap PSSI dan Kebijakan Pemain**  
Berisi postingan bernada kritik terhadap PSSI, perdebatan naturalisasi, dan kualitas pemain lokal. Kata dominan: `pssi`, `pemain`, `tapi`, `kalau`.

**Cluster 1 - Diskusi Umum Seputar Timnas Indonesia**  
Pembahasan umum tentang Timnas Indonesia di kompetisi FIFA, dengan beberapa noise seperti "sekolah garuda" dan "patung garuda". Kata dominan: `timnas`, `indonesia`, `fifa`, `cup`.

**Cluster 2 - #TimnasDay dan #PertiwiBerani**  
Postingan dengan tagar kampanye #TimnasDay dan #PertiwiBerani, termasuk dukungan Timnas Putri dan mention tokoh politik Gibran. Kata dominan: `timnasday`, `gibran`, `pertiwiberani`.
""")

    with col_kanan:
        with st.expander("Kemungkinan Tumpang Tindih", expanded=False):
            st.markdown("""
Cluster 0 dan 1 tumpang tindih pada kata "timnas" dan "pemain", perbedaannya terletak pada nada (cluster 0 lebih kritik, cluster 1 lebih informatif).

Cluster 1 dan 2 tumpang tindih pada kata "timnas" dan "garuda", cluster 2 lebih spesifik ke tagar kampanye, sedangkan cluster 1 lebih luas.

Kata "garuda" yang ambigu (bisa berarti julukan Timnas, brand, atau nama tempat) menyebabkan tumpang tindih antar cluster.
""")