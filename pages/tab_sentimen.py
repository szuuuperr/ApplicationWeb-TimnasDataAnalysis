
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from transformers import pipeline

# Lokasi folder proyek
BASE_DIR = Path(__file__).resolve().parent.parent
EVALUASI_DIR = BASE_DIR / "data" / "processed"
EVALUASI_DIR.mkdir(parents=True, exist_ok=True)

FILE_SAMPEL = EVALUASI_DIR / "sampel_evaluasi_sentimen.csv"
FILE_LABEL = EVALUASI_DIR / "label_manual_sentimen.csv"


@st.cache_resource
def load_model():
    """Memuat model sentimen satu kali selama aplikasi berjalan."""
    return pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier",
        device=-1
    )


def cari_kolom_teks(df):
    """Mencari kolom teks yang tersedia pada dataset."""
    kandidat = ["text", "full_text", "tweet", "teks", "text_clean"]

    for kolom in kandidat:
        if kolom in df.columns:
            return kolom

    return None


@st.cache_data(show_spinner=False)
def klasifikasi_sentimen(teks_list):
    """Mengklasifikasikan kumpulan postingan."""
    model = load_model()
    hasil = []

    # Batch kecil agar lebih ramah terhadap memori
    for i in range(0, len(teks_list), 16):
        batch = teks_list[i:i + 16]
        prediksi_batch = model(batch, truncation=True, batch_size=16)

        for prediksi in prediksi_batch:
            label_teratas = max(
                prediksi,
                key=lambda item: item["score"]
            )

            hasil.append({
                "sentimen": label_teratas["label"].lower(),
                "confidence": float(label_teratas["score"])
            })

    return pd.DataFrame(hasil)


def tampilkan_tab_sentimen(df):
    st.header("Analisis Sentimen Postingan X")
    st.write(
        "Klasifikasi sentimen postingan tentang Timnas Indonesia "
        "menggunakan model pretrained Bahasa Indonesia."
    )

    kolom_teks = cari_kolom_teks(df)

    if kolom_teks is None:
        st.error(
            "Kolom teks tidak ditemukan. Dataset harus memiliki "
            "kolom text, full_text, tweet, atau teks."
        )
        return

    # Menyiapkan data
    data = df.copy()
    data[kolom_teks] = data[kolom_teks].fillna("").astype(str)
    data = data[data[kolom_teks].str.strip() != ""].copy()
    data = data.reset_index(drop=True)

    if data.empty:
        st.warning("Tidak ada postingan yang dapat dianalisis.")
        return

    st.caption(f"Jumlah postingan yang dianalisis: {len(data)}")

    # Menjalankan klasifikasi
    with st.spinner("Model sedang menganalisis sentimen postingan..."):
        teks_list = tuple(data[kolom_teks].tolist())
        hasil = klasifikasi_sentimen(teks_list)

    data["sentimen"] = hasil["sentimen"].values
    data["confidence"] = hasil["confidence"].values

    # Ringkasan jumlah dan persentase
    urutan = ["positive", "neutral", "negative"]
    nama_label = {
        "positive": "Positif",
        "neutral": "Netral",
        "negative": "Negatif"
    }

    jumlah = data["sentimen"].value_counts()
    ringkasan = pd.DataFrame({
        "Sentimen": urutan,
        "Jumlah": [int(jumlah.get(label, 0)) for label in urutan]
    })
    ringkasan["Persentase"] = (
        ringkasan["Jumlah"] / len(data) * 100
    )
    ringkasan["Kategori"] = ringkasan["Sentimen"].map(nama_label)

    st.subheader("Distribusi Sentimen")

    kolom = st.columns(3)
    for i, label in enumerate(urutan):
        baris = ringkasan[ringkasan["Sentimen"] == label].iloc[0]
        with kolom[i]:
            st.metric(
                nama_label[label],
                f"{int(baris['Jumlah'])} postingan",
                f"{baris['Persentase']:.2f}%"
            )

    grafik = alt.Chart(ringkasan).mark_bar().encode(
        x=alt.X("Kategori:N", title="Kategori Sentimen", sort=[
            "Positif", "Netral", "Negatif"
        ]),
        y=alt.Y("Jumlah:Q", title="Jumlah Postingan"),
        tooltip=[
            alt.Tooltip("Kategori:N", title="Sentimen"),
            alt.Tooltip("Jumlah:Q", title="Jumlah"),
            alt.Tooltip("Persentase:Q", title="Persentase", format=".2f")
        ]
    ).properties(height=350)

    st.altair_chart(grafik, use_container_width=True)

    # Tabel hasil klasifikasi
    st.subheader("Hasil Klasifikasi Postingan")

    data_tampil = data.copy()
    data_tampil["sentimen"] = data_tampil["sentimen"].map(nama_label)
    data_tampil["confidence"] = (
        data_tampil["confidence"] * 100
    ).round(2)

    st.dataframe(
        data_tampil,
        use_container_width=True,
        hide_index=True
    )

    # Evaluasi manual
    st.divider()
    st.subheader("Evaluasi Manual 20 Postingan")
    st.write(
        "Ambil sampel acak untuk diberi label manual oleh anggota kelompok. "
        "File sampel tidak menyertakan prediksi model agar penilaian "
        "manual dapat dilakukan secara independen."
    )

    if st.button("Buat sampel evaluasi 20 postingan"):
        if len(data) < 20:
            st.warning(
                f"Dataset hanya memiliki {len(data)} postingan. "
                "Seluruh postingan akan digunakan sebagai sampel."
            )

        jumlah_sampel = min(20, len(data))
        sampel = data.sample(
            n=jumlah_sampel,
            random_state=None
        ).copy()

        sampel = sampel.reset_index(drop=True)
        sampel.insert(
            0, "sample_id",
            [f"S{i + 1:02d}" for i in range(len(sampel))]
        )

        # File untuk penilaian manual tidak memuat prediksi model
        sampel_manual = sampel[["sample_id", kolom_teks]].copy()
        sampel_manual = sampel_manual.rename(
            columns={kolom_teks: "text"}
        )
        sampel_manual["label_manual"] = ""

        sampel_manual.to_csv(
            FILE_SAMPEL,
            index=False,
            encoding="utf-8-sig"
        )

        # Simpan pasangan ID dan prediksi secara terpisah
        prediksi_sampel = sampel[["sample_id", "sentimen"]].copy()
        prediksi_sampel.to_csv(
            EVALUASI_DIR / "prediksi_sampel_internal.csv",
            index=False,
            encoding="utf-8-sig"
        )

        st.success(
            "Sampel evaluasi dibuat. Unduh CSV di bawah dan isi label "
            "manual tanpa melihat prediksi model."
        )

    if FILE_SAMPEL.exists():
        st.download_button(
            "Unduh CSV untuk label manual",
            data=FILE_SAMPEL.read_bytes(),
            file_name="sampel_evaluasi_sentimen.csv",
            mime="text/csv"
        )

    st.info(
        "Isi kolom label_manual dengan positive, neutral, atau negative. "
        "Simpan CSV hasil pelabelan, lalu unggah kembali untuk menghitung "
        "kesesuaian dengan prediksi model."
    )

    file_label = st.file_uploader(
        "Unggah CSV yang sudah diberi label manual",
        type=["csv"],
        key="unggah_label_manual"
    )

    if file_label is not None:
        try:
            manual = pd.read_csv(file_label)

            kolom_wajib = {"sample_id", "label_manual"}
            if not kolom_wajib.issubset(manual.columns):
                st.error(
                    "CSV harus memiliki kolom sample_id dan label_manual."
                )
                return

            manual["label_manual"] = (
                manual["label_manual"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            label_valid = {"positive", "neutral", "negative"}

            if not manual["label_manual"].isin(label_valid).all():
                st.error(
                    "Pastikan semua label manual berisi positive, "
                    "neutral, atau negative."
                )
                return

            file_prediksi = (
                EVALUASI_DIR / "prediksi_sampel_internal.csv"
            )

            if not file_prediksi.exists():
                st.error(
                    "File prediksi internal tidak ditemukan. "
                    "Buat sampel evaluasi terlebih dahulu."
                )
                return

            prediksi = pd.read_csv(file_prediksi)

            perbandingan = manual.merge(
                prediksi,
                on="sample_id",
                how="inner",
                validate="one_to_one"
            )

            if len(perbandingan) != len(manual):
                st.error(
                    "Ada sample_id yang tidak cocok. Gunakan CSV sampel "
                    "yang diunduh dari sesi evaluasi yang sama."
                )
                return

            perbandingan = perbandingan.rename(
                columns={"sentimen": "prediksi_model"}
            )
            perbandingan["sesuai"] = (
                perbandingan["label_manual"]
                == perbandingan["prediksi_model"]
            )

            jumlah_sesuai = int(perbandingan["sesuai"].sum())
            total = len(perbandingan)
            persentase = jumlah_sesuai / total * 100

            st.subheader("Hasil Evaluasi")
            c1, c2, c3 = st.columns(3)
            c1.metric("Sampel dievaluasi", total)
            c2.metric("Label sesuai", jumlah_sesuai)
            c3.metric("Persentase kesesuaian", f"{persentase:.2f}%")

            st.caption(
                "Persentase ini adalah kesesuaian pada sampel manual, "
                "bukan ukuran akurasi yang representatif untuk seluruh "
                "postingan X."
            )

            berbeda = perbandingan[
                ~perbandingan["sesuai"]
            ].copy()

            st.subheader("Postingan dengan Label Berbeda")
            st.write(
                f"Ditemukan {len(berbeda)} perbedaan. "
                "Pilih hingga tiga contoh untuk dibahas dalam laporan."
            )

            if berbeda.empty:
                st.success(
                    "Semua label manual pada sampel ini sesuai dengan "
                    "prediksi model."
                )
            else:
                st.dataframe(
                    berbeda[
                        [
                            "sample_id",
                            "label_manual",
                            "prediksi_model"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                # Tampilkan teks untuk membantu pembahasan
                teks_manual = manual[
                    ["sample_id", "text"]
                ]
                contoh = berbeda.merge(
                    teks_manual,
                    on="sample_id",
                    how="left"
                ).head(3)

                for _, baris in contoh.iterrows():
                    with st.expander(
                        f"{baris['sample_id']}: "
                        f"{baris['label_manual']} vs "
                        f"{baris['prediksi_model']}"
                    ):
                        st.write(baris["text"])

        except Exception as e:
            st.error(f"Gagal memproses CSV evaluasi: {e}")
