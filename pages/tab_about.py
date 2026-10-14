import streamlit as st


def tampilkan_tab_about(df=None):
    st.header("Tentang & Temuan Utama")

    jumlah_data = len(df) if df is not None else 0
    jumlah_data_teks = f"{jumlah_data:,}".replace(",", ".")

    st.write(
        "Aplikasi ini menganalisis postingan X tentang Tim Nasional Indonesia "
        "melalui empat pendekatan: pembersihan data, analisis sentimen, clustering, "
        "dan regresi. Ringkasan hasil beserta keterbatasannya dijelaskan di bawah ini."
    )

    st.divider()
    st.subheader("Temuan Utama")
    st.caption(f"Berdasarkan {jumlah_data_teks} postingan yang lolos tahap pembersihan.")

    with st.container(border=True):
        st.markdown("#### 1. Percakapan didominasi sentimen negatif dan netral")
        c1, c2, c3 = st.columns(3)
        c1.metric("Negatif", "40,7%")
        c2.metric("Netral", "40,4%")
        c3.metric("Positif", "18,9%")
        st.markdown(
            "Model sentimen bahasa Indonesia menempatkan mayoritas postingan pada "
            "kategori **negatif dan netral**. Dukungan positif justru menjadi minoritas. "
            "Percakapan publik tentang Timnas terbelah, sejalan dengan ramainya kritik "
            "terhadap federasi, pemilihan pelatih, dan dinamika pemain."
        )

    with st.container(border=True):
        st.markdown("#### 2. Percakapan terpolarisasi ke dalam tiga tema besar")
        st.markdown(
            "- **FIFA ASEAN Cup 2026 & Prestasi Timnas**, sekitar 46 postingan, berisi "
            "euforia kemenangan, trofi, dan pencapaian skuad.\n"
            "- **Dinamika PSSI, Pemain & Pelatih**, sekitar 378 postingan, berisi "
            "diskusi umum yang bercampur kritik terhadap kebijakan federasi dan "
            "kualitas pemain.\n"
            "- **#TimnasDay dan #PertiwiBerani**, sekitar 121 postingan, berisi tagar "
            "kampanye dan dukungan suporter yang membentuk komunitas percakapan tersendiri."
        )
        st.markdown(
            "Hasil K-Means + TF-IDF menunjukkan tagar kampanye mampu mengelompokkan "
            "percakapan secara mandiri, terpisah dari diskusi umum maupun euforia gelar."
        )

    with st.container(border=True):
        st.markdown("#### 3. Jumlah like sangat kuat memprediksi jumlah repost")
        c1, c2, c3 = st.columns(3)
        c1.metric("Korelasi Pearson", "0,9954")
        c2.metric("R\u00b2", "0,9932")
        c3.metric("MAE", "2,74")
        st.markdown(
            "Model regresi linear `repost = 0,10 x like + 0,22` menjelaskan sekitar "
            "**99,3%** variasi jumlah repost. Keterlibatan *like* dan *repost* bergerak "
            "hampir seiring: postingan yang banyak disukai cenderung disebar ulang. "
            "Sebagai catatan, variabel awal berupa `jumlah_kata` dan `jumlah_hashtag` "
            "tidak dipakai karena korelasinya terhadap interaksi jauh lebih lemah."
        )

    st.divider()
    st.subheader("Keterbatasan")

    with st.container(border=True):
        st.markdown("#### 1. Keterbatasan data")
        st.markdown(
            f"- Sampel hanya **{jumlah_data_teks} postingan** yang diambil dari "
            "sejumlah kata kunci terbatas dalam rentang waktu tertentu, sehingga "
            "**tidak mewakili** seluruh opini masyarakat Indonesia.\n"
            "- Distribusi interaksi sangat **miring**: sedikit postingan viral dengan "
            "like/repost sangat besar, sementara mayoritas bernilai kecil.\n"
            "- Kata kunci **\"garuda\" bersifat ambigu** karena dapat merujuk pada "
            "maskapai, brand, maupun nama tempat, sehingga berpotensi menyisakan noise "
            "meski sudah difilter."
        )

    with st.container(border=True):
        st.markdown("#### 2. Keterbatasan metode")
        st.markdown(
            "- **Regresi** masih linear sederhana dengan satu fitur berupa jumlah "
            "*like*, dan mengabaikan usia postingan, jangkauan akun, serta waktu "
            "publikasi.\n"
            "- **Clustering** memakai jumlah cluster tetap sebanyak tiga tanpa validasi "
            "*elbow* maupun *silhouette*, dan hasilnya sensitif terhadap preprocessing.\n"
            "- **Sentimen** bergantung pada model pra-latih pihak ketiga; selain itu "
            "teks `text_clean` menghapus emoji dan tanda baca yang dapat membawa nuansa "
            "sentimen, sehingga akurasi berpotensi menurun."
        )
