import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def tampilkan_tab_regresi(data):
    st.header("Regresi Linear - Prediksi Likes")

    text_col = 'text' if 'text' in data.columns else 'full_text'
    like_col = 'like_count' if 'like_count' in data.columns else None

    if like_col is None:
        st.error("Kolom `like_count` tidak ditemukan di dataset.")
        return

    df = data.dropna(subset=[like_col]).copy()
    df = df[df[like_col].astype(str).str.strip() != '']
    df[like_col] = pd.to_numeric(df[like_col], errors='coerce')
    df = df.dropna(subset=[like_col]).reset_index(drop=True)

    jumlah_awal = len(data)
    jumlah_valid = len(df)
    jumlah_dibuang = jumlah_awal - jumlah_valid
    st.caption(f"Data awal: **{jumlah_awal}** baris | "
               f"Valid: **{jumlah_valid}** baris | "
               f"Dibuang (like_count invalid): **{jumlah_dibuang}** baris")

    if len(df) < 10:
        st.warning("Data terlalu sedikit untuk regresi (minimal 10 baris).")
        return

    df['jumlah_kata'] = df[text_col].astype(str).str.split().str.len()
    df['jumlah_hashtag'] = df[text_col].astype(str).apply(lambda t: len(re.findall(r'#\w+', t)))

    X = df[['jumlah_kata', 'jumlah_hashtag']]
    y = df[like_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.caption(f"Data valid: **{len(df)}** baris | "
               f"Latih: **{len(X_train)}** | Uji: **{len(X_test)}**")

    st.subheader("Evaluasi Model")
    col1, col2 = st.columns(2)
    col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    col2.metric("R² (Koefisien Determinasi)", f"{r2:.4f}")

    with st.expander("Detail koefisien model"):
        st.write(f"**Intercept**: {model.intercept_:.2f}")
        st.write(f"**Koefisien jumlah_kata**: {model.coef_[0]:.4f}")
        st.write(f"**Koefisien jumlah_hashtag**: {model.coef_[1]:.4f}")

    st.divider()

    st.subheader("Estimasi Likes")

    col1, col2, col3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    kata = col1.number_input("Jumlah kata", min_value=0, value=20, step=1)
    hashtag = col2.number_input("Jumlah hashtag", min_value=0, value=2, step=1)
    tombol = col3.button("Prediksi", use_container_width=True)

    if tombol:
        input_data = pd.DataFrame({'jumlah_kata': [kata], 'jumlah_hashtag': [hashtag]})
        pred = model.predict(input_data)[0]
        if pred < 0:
            st.warning(f"Estimasi: **{pred:.2f}** - nilai negatif tidak bermakna "
                       f"sebagai jumlah likes (keterbatasan model).")
        else:
            st.success(f"Estimasi likes untuk **{kata} kata** dan **{hashtag} hashtag**: "
                       f"**{pred:.2f} likes**")

    st.divider()

    with st.expander("Likes Aktual vs Prediksi", expanded=False):
        st.caption("Klik icon fullscreen di kanan atas grafik untuk memperbesar")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=20)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', label='Garis ideal')
        ax.set_xlabel('Likes Aktual', fontsize=9)
        ax.set_ylabel('Likes Prediksi', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Exploratory Data Analysis", expanded=False):

        st.markdown("**1. Distribusi Data**")
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        sns.histplot(df['jumlah_kata'], bins=20, kde=True, ax=axes[0], color='steelblue')
        axes[0].set_title('Jumlah Kata', fontsize=9)
        sns.histplot(df['jumlah_hashtag'], bins=20, kde=True, ax=axes[1], color='orange')
        axes[1].set_title('Jumlah Hashtag', fontsize=9)
        sns.histplot(df[like_col], bins=20, kde=True, ax=axes[2], color='green')
        axes[2].set_title('Likes', fontsize=9)
        for ax in axes:
            ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**2. Hubungan Antar Variabel**")
        pair_fig = sns.pairplot(df[['jumlah_kata', 'jumlah_hashtag', like_col]],
                                diag_kind='kde', height=1.6,
                                plot_kws={'s': 10, 'alpha': 0.5})
        pair_fig.fig.suptitle('Hubungan Antar Variabel', y=1.02, fontsize=10)
        st.pyplot(pair_fig.fig)

        st.markdown("**3. Matriks Korelasi**")
        corr = df[['jumlah_kata', 'jumlah_hashtag', like_col]].corr()
        col_corr1, col_corr2 = st.columns([1, 2])
        with col_corr1:
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f',
                        vmin=-1, vmax=1, square=True, ax=ax,
                        annot_kws={'size': 8})
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
        with col_corr2:
            st.write("Nilai korelasi:")
            st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

        st.markdown("**4. Grafik Regresi (Fitur vs Likes)**")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        sns.regplot(data=df, x='jumlah_kata', y=like_col,
                    ax=axes[0], scatter_kws={'alpha': 0.4, 's': 10},
                    line_kws={'color': 'red'})
        axes[0].set_title('Jumlah Kata vs Likes', fontsize=9)
        sns.regplot(data=df, x='jumlah_hashtag', y=like_col,
                    ax=axes[1], scatter_kws={'alpha': 0.4, 's': 10},
                    line_kws={'color': 'red'})
        axes[1].set_title('Jumlah Hashtag vs Likes', fontsize=9)
        for ax in axes:
            ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    st.subheader("Interpretasi")

    if r2 < 0:
        r2_text = (f"Nilai R2 negatif ({r2:.4f}) menunjukkan model **lebih buruk** "
                   f"daripada hanya memprediksi nilai rata-rata likes.")
    else:
        r2_text = f"Model menjelaskan sekitar **{r2*100:.1f}%** variasi jumlah likes."

    st.markdown(f"""
- **MAE ({mae:.2f})**: rata-rata kesalahan prediksi sekitar **+/-{mae:.0f} likes**.
- **R2 ({r2:.4f})**: {r2_text}
- **Keterbatasan**: model hanya menggunakan 2 fitur (jumlah kata & jumlah hashtag).
  Jumlah likes juga dipengaruhi oleh usia postingan, jangkauan akun, isu yang sedang
  trending, dan waktu publikasi.
""")