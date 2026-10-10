# Analisis Sentimen, Clustering, dan Regresi Likes pada Postingan X tentang Timnas

Tugas mata kuliah Aplikasi Web, Program Studi Teknologi Informasi, Universitas Negeri Yogyakarta.

Dosen pengampu: Akhsin Nurlayli S.Pd., M.Eng.

Anggota kelompok:

| Nama | NIM |
|---|---|
| Muhammad Yafi Adyatma | 24051130006 |
| Tsabitah Sarah | 24051130031 |
| M. Arrizal Rahman A.L. | 24051130040 |

## Deskripsi

Aplikasi web berbasis Streamlit untuk menganalisis postingan X tentang Tim Nasional Indonesia. Aplikasi memiliki empat tab: Data, Sentimen, Clustering, dan Regresi. Aplikasi hanya membaca dataset CSV dan tidak mengambil data langsung dari X.

## Struktur Berkas

```
├── app.py
├── requirements.txt
├── README.md
│
├── pages/
│   ├── tab_data.py
│   ├── tab_sentimen.py
│   ├── tab_clustering.py
│   └── tab_regresi.py
│
├── notebooks/
│   └── 01_pengambilan_pembersihan_data.ipynb
│
├── data/
│   ├── raw/
│   │   └── dataset_timnas_mentah.csv
│   │
│   └── processed/
│       ├── dataset_timnas.csv
│       ├── metadata_pengambilan.json
│       └── cleaning_log.csv
│
├── evaluasi/
│   └── evaluasi_sentimen.csv
│
└── .gitignore
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
| text_clean | Teks hasil pembersihan untuk sentimen dan clustering |

Kolom wajib aplikasi adalah `text`, `created_at`, dan `like_count`.

## Tahap Pembersihan Data

1. Hapus teks kosong dan nilai hilang.
2. Hapus duplikat berdasarkan teks asli.
3. Buang postingan yang tidak relevan dengan topik timnas.
4. Buat `text_clean` dengan huruf kecil, tanpa URL, mention, dan simbol. Kata negasi dipertahankan.
5. Hapus duplikat berdasarkan teks bersih.
6. Validasi `like_count`. Nilai tidak valid dibiarkan kosong dan tidak diganti dengan nol.

Jumlah baris di setiap tahap tersimpan pada `cleaning_log.csv` dan ditampilkan di tab Data.

## Batasan

- Hasil hanya berlaku untuk sampel postingan yang dikumpulkan dan tidak mewakili seluruh opini masyarakat.
- Jumlah likes dipengaruhi faktor lain seperti usia postingan dan jangkauan akun.
