from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

# Path berkas sesuai struktur proyek
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILE_EVALUASI = PROCESSED_DIR / "evaluasi_sentimen.csv"
FILE_PREDIKSI_TEMP = PROCESSED_DIR / "prediksi_sampel_internal.csv"

URUTAN_SENTIMEN = ["positive", "neutral", "negative"]
NAMA_LABEL = {"positive": "Positif", "neutral": "Netral", "negative": "Negatif"}

# Kata Kunci Sentimen untuk Fallback Lexicon
KATA_POSITIF = {
    "menang", "juara", "mantap", "bagus", "keren", "hebat", "bangga", "terbaik", 
    "garuda", "dukung", "semangat", "gacor", "top", "apresiasi", "lolos", "unggul",
    "terimakasih", "berhasil", "jos", "menyala", "kelas", "cinta", "love", "pinter"
}

KATA_NEGATIF = {
    "kalah", "jelek", "buruk", "kecewa", "parah", "lemah", "bobrok", "out", 
    "gagal", "rugi", "ancur", "hancur", "error", "salah", "benci", "ampas", 
    "beban", "boikot", "injak", "kehilangan", "rusak"
}

KATA_NEGASI = {"tidak", "bukan", "gak", "nggak", "ngga", "tak", "belum", "tdk"}


def analisis_sentimen_lexicon(teks):
    """Fallback analisis sentimen berbasis aturan & negasi."""
    kata_kata = str(teks).lower().split()
    skor = 0
    for i, kata in enumerate(kata_kata):
        is_negasi = (i > 0 and kata_kata[i - 1] in KATA_NEGASI) or (i > 1 and kata_kata[i - 2] in KATA_NEGASI)
        if kata in KATA_POSITIF:
            skor += -1 if is_negasi else 1
        elif kata in KATA_NEGATIF:
            skor += 1 if is_negasi else -1

    if skor > 0:
        return "positive", min(0.55 + skor * 0.1, 0.98)
    elif skor < 0:
        return "negative", min(0.55 + abs(skor) * 0.1, 0.98)
    return "neutral", 0.60


@st.cache_resource
def muat_model_transformer():
    """Memuat model HuggingFace bila tersedia."""
    try:
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            device=-1
        )
    except Exception:
        return None


def cari_kolom_teks(df):
    for k in ["text_clean", "text", "full_text", "tweet", "teks"]:
        if k in df.columns:
            return k
    return None


@st.cache_data(show_spinner=False)
def proses_klasifikasi(teks_list):
    """Proses prediksi sentimen dengan transformer atau fallback lexicon."""
    model = muat_model_transformer()
    hasil = []

    if model is not None:
        try:
            batch_size = 16
            for i in range(0, len(teks_list), batch_size):
                batch = list(teks_list[i:i + batch_size])
                preds = model(batch, truncation=True, batch_size=batch_size)
                for p in preds:
                    item = p[0] if isinstance(p, list) else p
                    raw_label = str(item.get("label", "")).strip().lower()
                    score = float(item.get("score", 0.6))

                    # Normalisasi label
                    if "pos" in raw_label or raw_label == "label_2":
                        label = "positive"
                    elif "neg" in raw_label or raw_label == "label_0":
                        label = "negative"
                    else:
                        label = "neutral"

                    hasil.append({"sentimen": label, "confidence": score})
            return pd.DataFrame(hasil)
        except Exception:
            pass

    # Menggunakan Lexicon jika HuggingFace tidak dapat digunakan
    for t in teks_list:
        lbl, conf = analisis_sentimen_lexicon(t)
        hasil.append({"sentimen": lbl, "confidence": conf})
    return pd.DataFrame(hasil)


def tampilkan_tab_sentimen(df):
    st.header("Analisis Sentimen Postingan X")
    st.write("Klasifikasi sentimen postingan tentang Timnas Indonesia ke dalam kategori Positif, Netral, dan Negatif.")

    kolom_teks = cari_kolom_teks(df)
    if not kolom_teks:
        st.error("Kolom teks tidak ditemukan pada dataset.")
        return

    data = df.copy()
    data[kolom_teks] = data[kolom_teks].fillna("").astype(str)
    data = data[data[kolom_teks].str.strip() != ""].reset_index(drop=True)

    if data.empty:
        st.warning("Tidak ada data teks yang dapat dianalisis.")
        return

    # Menjalankan Klasifikasi
    with st.spinner("Menganalisis sentimen postingan..."):
        teks_tuple = tuple(data[kolom_teks].tolist())
        hasil_pred = proses_klasifikasi(teks_tuple)

    data["sentimen"] = hasil_pred["sentimen"].values
    data["confidence"] = hasil_pred["confidence"].values

    # 1. Ringkasan & Persentase
    jumlah = data["sentimen"].value_counts()
    ringkasan = pd.DataFrame({
        "Sentimen": URUTAN_SENTIMEN,
        "Jumlah": [int(jumlah.get(lbl, 0)) for lbl in URUTAN_SENTIMEN]
    })
    ringkasan["Persentase"] = (ringkasan["Jumlah"] / len(data)) * 100
    ringkasan["Kategori"] = ringkasan["Sentimen"].map(NAMA_LABEL)

    st.subheader("Distribusi Sentimen")
    cols = st.columns(3)
    for i, lbl in enumerate(URUTAN_SENTIMEN):
        row = ringkasan[ringkasan["Sentimen"] == lbl].iloc[0]
        cols[i].metric(NAMA_LABEL[lbl], f"{row['Jumlah']} postingan", f"{row['Persentase']:.2f}%")

    # 2. Grafik Distribusi Sentimen
    chart = alt.Chart(ringkasan).mark_bar().encode(
        x=alt.X("Kategori:N", title="Kategori Sentimen", sort=["Positif", "Netral", "Negatif"]),
        y=alt.Y("Jumlah:Q", title="Jumlah Postingan"),
        color=alt.Color("Kategori:N", legend=None),
        tooltip=["Kategori", "Jumlah", alt.Tooltip("Persentase:Q", format=".2f")]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    # 3. Tabel Hasil Klasifikasi
    st.subheader("Tabel Hasil Klasifikasi")
    tampil_df = data.copy()
    tampil_df["sentimen_label"] = tampil_df["sentimen"].map(NAMA_LABEL)
    tampil_df["confidence_pct"] = (tampil_df["confidence"] * 100).round(2)

    st.dataframe(
        tampil_df[["text", "sentimen_label", "confidence_pct"]],
        use_container_width=True,
        hide_index=True
    )

    # 4. Pemeriksaan Manual 20 Postingan
    st.divider()
    st.subheader("Pemeriksaan Manual 20 Sampel (Evaluasi Model)")

    # Jika file evaluasi_sentimen.csv sudah ada di folder processed
    # Jika file evaluasi_sentimen.csv sudah ada di folder processed
    if FILE_EVALUASI.exists():
        eval_df = pd.read_csv(FILE_EVALUASI)
        if {"text", "label_manual", "prediksi_model"}.issubset(eval_df.columns):
            # Pastikan kolom diubah ke tipe string terlebih dahulu
            label_manual_str = eval_df["label_manual"].fillna("").astype(str).str.lower().str.strip()
            prediksi_model_str = eval_df["prediksi_model"].fillna("").astype(str).str.lower().str.strip()
            
            eval_df["sesuai"] = label_manual_str == prediksi_model_str
            total_eval = len(eval_df)
            jumlah_sesuai = int(eval_df["sesuai"].sum())
            persen_sesuai = (jumlah_sesuai / total_eval * 100) if total_eval > 0 else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sampel Evaluasi", total_eval)
            m2.metric("Jumlah Label Sesuai", jumlah_sesuai)
            m3.metric("Persentase Kesesuaian", f"{persen_sesuai:.2f}%")

            # Pembahasan 3 Contoh Ketidaksesuaian
            mismatch_df = eval_df[~eval_df["sesuai"]].copy()
            st.write(f"**Ditemukan {len(mismatch_df)} ketidaksesuaian label.**")

            if not mismatch_df.empty:
                st.write("**Contoh Ketidaksesuaian untuk Laporan:**")
                contoh_3 = mismatch_df.head(3)
                for _, idx_row in contoh_3.iterrows():
                    with st.expander(f"Sampel {idx_row.get('sample_id', '')} | Manual: {idx_row['label_manual']} VS Sistem: {idx_row['prediksi_model']}"):
                        st.write(f"**Teks:** {idx_row['text']}")

    # Pembuatan Sampel Baru
    st.write("---")
    st.write("  **Alur Pembuatan / Unggah Evaluasi Manual**")
    if st.button("Buat / Acak 20 Sampel Baru", key="btn_sample"):
        n_sample = min(20, len(data))
        sampel = data.sample(n=n_sample, random_state=42).reset_index(drop=True)
        sampel["sample_id"] = [f"S{i+1:02d}" for i in range(len(sampel))]

        # Simpan file prediksi internal
        sampel[["sample_id", "text", "sentimen"]].rename(columns={"sentimen": "prediksi_model"}).to_csv(
            FILE_PREDIKSI_TEMP, index=False, encoding="utf-8-sig"
        )

        # File sampel kosong untuk diisi manual
        sampel_manual = sampel[["sample_id", "text"]].copy()
        sampel_manual["label_manual"] = ""
        sampel_manual.to_csv(FILE_EVALUASI, index=False, encoding="utf-8-sig")
        st.success("20 sampel berhasil diacak dan disimpan!")

    uploaded_eval = st.file_uploader("Unggah `evaluasi_sentimen.csv` yang telah diisi label_manual", type=["csv"])
    if uploaded_eval is not None:
        try:
            up_df = pd.read_csv(uploaded_eval)
            if "label_manual" in up_df.columns:
                if FILE_PREDIKSI_TEMP.exists():
                    pred_temp = pd.read_csv(FILE_PREDIKSI_TEMP)
                    up_df = up_df.merge(pred_temp[["sample_id", "prediksi_model"]], on="sample_id", how="left")
                up_df.to_csv(FILE_EVALUASI, index=False, encoding="utf-8-sig")
                st.success("Berkas evaluasi berhasil diperbarui!")
                st.rerun()
        except Exception as err:
            st.error(f"Gagal memproses berkas: {err}")