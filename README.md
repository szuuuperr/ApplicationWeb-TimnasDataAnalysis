# Analisis Sentimen, Clustering, dan Regresi Repost pada Postingan X tentang Timnas

Tugas mata kuliah Aplikasi Web, Program Studi Teknologi Informasi, Universitas Negeri Yogyakarta.

Dosen pengampu: Akhsin Nurlayli S.Pd., M.Eng.

Anggota kelompok:

| Nama | NIM |
|---|---|
| Muhammad Yafi Adyatma | 24051130006 |
| Tsabitah Sarah | 24051130031 |
| M. Arrizal Rahman A.L. | 24051130040 |

## Deskripsi

Aplikasi web berbasis Streamlit untuk menganalisis postingan X tentang Tim Nasional Indonesia. Aplikasi memiliki lima tab: Data, Sentimen, Clustering, Regresi, dan About. Aplikasi hanya membaca dataset CSV dan tidak mengambil data langsung dari X.

- **Data** menampilkan ringkasan dataset, log tahap pembersihan, dan metadata pengambilan data.
- **Sentimen** mengklasifikasikan postingan menjadi Positif, Netral, dan Negatif.
- **Clustering** mengelompokkan postingan dengan TF-IDF + K-Means.
- **Regresi** memprediksi jumlah repost berdasarkan jumlah like.
- **About** merangkum temuan utama dan keterbatasan hasil analisis.

## Struktur Berkas

```
app.py
requirements.txt
README.md
pages/
    tab_data.py
    tab_sentimen.py
    tab_clustering.py
    tab_regresi.py
    tab_about.py
notebooks/
    01_pengambilan_pembersihan_data.ipynb
    02_clustering_timnas.ipynb
data/
    raw/
        dataset_timnas_mentah.csv
        crawl/
            garuda.csv
            pssi.csv
            timnas.csv
            timnasday.csv
            timnas_indonesia.csv
    processed/
        dataset_timnas.csv
        metadata_pengambilan.json
        cleaning_log.csv
        evaluasi_sentimen.csv
        prediksi_sampel_internal.csv
.gitignore
```

## Cara Menjalankan

1. Pasang Python 3.10 atau lebih baru.
2. Buat dan aktifkan virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. Pasang dependensi.

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi.

   ```bash
   streamlit run app.py
   ```

Aplikasi akan membaca `data/processed/dataset_timnas.csv` secara otomatis. File CSV lain dapat diunggah melalui tombol unggah di bagian atas aplikasi.

## Menjalankan Ulang Pengambilan Data

Langkah ini opsional jika `data/processed/dataset_timnas.csv` sudah tersedia.

1. Pasang Node.js versi 18 atau lebih baru.
2. Ambil auth token dari cookie akun X yang sedang login.
3. Simpan token sebagai environment variable. Jangan menulis token di notebook.

   ```bash
   export X_AUTH_TOKEN="isi_token_anda"      # Windows PowerShell: $env:X_AUTH_TOKEN="isi_token_anda"
   ```

4. Buka `notebooks/01_pengambilan_pembersihan_data.ipynb` dan jalankan seluruh sel. Jalankan notebook dari folder `notebooks/` karena path bersifat relatif.

Jika crawling tidak memungkinkan, letakkan dataset cadangan yang disetujui dosen di `data/raw/dataset_timnas_mentah.csv`, lalu ubah `RUN_CRAWL` menjadi `False` pada notebook.

## Format Dataset

| Kolom | Keterangan |
|---|---|
| text | Teks asli postingan |
| created_at | Waktu publikasi dalam WIB |
| like_count | Jumlah likes saat data diambil |
| repost_count | Jumlah repost, jika tersedia |
| reply_count | Jumlah balasan, jika tersedia |
| url | URL postingan, jika tersedia |
| kata_kunci | Kata kunci yang menghasilkan postingan |
| lang | Kode bahasa postingan hasil deteksi |
| text_clean | Teks hasil pembersihan untuk sentimen dan clustering |

Kolom wajib aplikasi adalah `text`, `created_at`, dan `like_count`. Tab Regresi memerlukan tambahan kolom `repost_count`; bila tidak tersedia, tab tersebut menampilkan pesan dan tidak diproses.

## Tahap Pembersihan Data

1. Hapus teks kosong dan nilai hilang.
2. Hapus duplikat berdasarkan teks asli.
3. Buang postingan yang tidak relevan dengan topik timnas.
4. Buang postingan yang bukan berbahasa Indonesia.
5. Buat `text_clean` dengan huruf kecil, tanpa URL, mention, dan simbol. Kata negasi dipertahankan.
6. Hapus duplikat berdasarkan teks bersih.
7. Validasi `like_count`. Nilai tidak valid dibiarkan kosong dan tidak diganti dengan nol.

Jumlah baris di setiap tahap tersimpan pada `cleaning_log.csv` dan ditampilkan di tab Data.

## Temuan Utama

1. **Percakapan didominasi sentimen negatif dan netral.** Dari klasifikasi model bahasa Indonesia, sebaran sentimen adalah Negatif 40,7%, Netral 40,4%, dan Positif 18,9%. Dukungan positif justru menjadi minoritas.
2. **Percakapan terpolarisasi ke dalam tiga tema besar.** K-Means + TF-IDF menghasilkan tema *FIFA ASEAN Cup 2026 & Prestasi Timnas* dengan sekitar 46 postingan, *Dinamika PSSI, Pemain & Pelatih* dengan sekitar 378 postingan, dan *#TimnasDay & #PertiwiBerani* dengan sekitar 121 postingan.
3. **Jumlah like sangat kuat memprediksi jumlah repost.** Regresi linear `repost = 0,10 x like + 0,22` menghasilkan korelasi Pearson 0,9954, R² 0,9932, dan MAE 2,74.

Rangkuman ini juga ditampilkan pada tab **About** di dalam aplikasi.

## Keterbatasan

**Data**

- Sampel hanya berasal dari sejumlah kata kunci terbatas dalam rentang waktu tertentu sehingga tidak mewakili seluruh opini masyarakat Indonesia.
- Distribusi interaksi sangat miring: sedikit postingan viral dengan like/repost sangat besar.
- Kata kunci "garuda" bersifat ambigu karena dapat merujuk pada maskapai, brand, maupun nama tempat, sehingga berpotensi menyisakan noise.

**Metode**

- Regresi masih linear sederhana dengan satu fitur berupa jumlah like dan mengabaikan usia postingan, jangkauan akun, serta waktu publikasi.
- Clustering memakai jumlah cluster tetap sebanyak tiga tanpa validasi *elbow* maupun *silhouette* dan sensitif terhadap preprocessing.
- Sentimen bergantung pada model pra-latih pihak ketiga, serta `text_clean` menghapus emoji dan tanda baca yang dapat membawa nuansa sentimen sehingga akurasi berpotensi menurun.
