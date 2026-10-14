import json
import os

import pandas as pd
import streamlit as st

KOLOM_WAJIB = ["text", "created_at", "like_count"]
METADATA_PATH = "data/processed/metadata_pengambilan.json"
LOG_PATH = "data/processed/cleaning_log.csv"

@st.cache_data
def muat_dataset(file_unggah=None, path_default="data/processed/dataset_timnas.csv"):
    """Membaca CSV dari unggahan atau berkas default. Mengembalikan None bila tidak ada."""
    try:
        if file_unggah is not None:
            return pd.read_csv(file_unggah)
        if os.path.exists(path_default):
            return pd.read_csv(path_default)
    except pd.errors.EmptyDataError:
        st.error("Berkas CSV kosong dan tidak dapat diproses.")
        st.stop()
    except Exception as e:
        st.error(f"Berkas CSV tidak dapat dibaca: {e}")
        st.stop()
    return None


def validasi_dataset(df):
    """Menghentikan aplikasi dengan pesan jelas bila data tidak layak diproses."""
    hilang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if hilang:
        st.error(f"Kolom wajib tidak tersedia: {', '.join(hilang)}. "
                 f"Kolom yang dibutuhkan: {', '.join(KOLOM_WAJIB)}.")
        st.stop()
    if df.empty:
        st.error("Dataset tidak memiliki baris data.")
        st.stop()
    if df["text"].astype(str).str.strip().eq("").all():
        st.error("Seluruh kolom text kosong sehingga data tidak dapat diproses.")
        st.stop()


def siapkan_dataset(df):
    """Memastikan tipe kolom benar dan text_clean tersedia."""
    df = df.copy()
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce")
    if "text_clean" not in df.columns:
        df["text_clean"] = (
            df["text"].astype(str).str.lower()
            .str.replace(r"http\S+|@\w+", " ", regex=True)
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip()
        )
    return df


def baca_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def tampilkan_tab_data(df):
    st.header("Data")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah postingan", len(df))
    c2.metric("Like valid", int(df["like_count"].notna().sum()))
    c3.metric("Like tidak valid", int(df["like_count"].isna().sum()))

    if len(df) < 150:
        st.warning("Jumlah postingan kurang dari target minimal 150 postingan unik.")

    if df["like_count"].isna().any():
        st.info("Baris dengan like_count tidak valid tidak diganti dengan nol. "
                "Baris tersebut hanya dikeluarkan dari analisis regresi.")

    st.subheader("Pratinjau dataset")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Informasi pengambilan data")
    meta = baca_metadata()
    if meta is None:
        st.info("Berkas metadata_pengambilan.json tidak ditemukan. "
                "Jalankan notebook pengambilan data terlebih dahulu.")
    else:
        baris_meta = []
        if meta.get("kata_kunci"):
            baris_meta.append(f"- **Kata kunci:** {', '.join(meta['kata_kunci'])}")
        if meta.get("metode"):
            baris_meta.append(f"- **Metode:** {meta['metode']}")
        if meta.get("waktu_pengambilan"):
            baris_meta.append(f"- **Waktu pengambilan:** {meta['waktu_pengambilan']}")
        if meta.get("jumlah_data_mentah") is not None:
            baris_meta.append(f"- **Jumlah data mentah:** {meta['jumlah_data_mentah']}")
        if meta.get("jumlah_data_akhir") is not None:
            baris_meta.append(f"- **Jumlah data akhir:** {meta['jumlah_data_akhir']}")
        st.markdown("\n".join(baris_meta))
        if meta.get("kendala"):
            st.markdown("**Kendala yang dicatat:**")
            for k in meta["kendala"]:
                st.markdown(f"- {k}")

    if os.path.exists(LOG_PATH):
        st.subheader("Ringkasan pembersihan data")
        st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True)
