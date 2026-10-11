import streamlit as st

from pages.tab_data import (muat_dataset, validasi_dataset, siapkan_dataset, tampilkan_tab_data)
from pages.tab_sentimen import tampilkan_tab_sentimen
from pages.tab_clustering import tampilkan_tab_clustering
from pages.tab_regresi import tampilkan_tab_regresi

st.set_page_config(page_title="Analisis Postingan X tentang Timnas", layout="wide")
st.title("Analisis Postingan X tentang Timnas Indonesia")

file_unggah = st.file_uploader("Unggah dataset CSV", type="csv")
df = muat_dataset(file_unggah)

if df is None:
    st.info("Unggah file CSV atau letakkan dataset_timnas.csv di folder data/processed.")
    st.stop()

validasi_dataset(df)
df = siapkan_dataset(df)

tab_data, tab_sentimen, tab_cluster, tab_regresi = st.tabs(
    ["Data", "Sentimen", "Clustering", "Regresi"]
)

with tab_data:
    tampilkan_tab_data(df)

with tab_sentimen:
    tampilkan_tab_sentimen(df)

with tab_cluster:
    tampilkan_tab_clustering(df)

with tab_regresi:
    tampilkan_tab_regresi(df)
