import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def tampilkan_tab_regresi(data):
    st.header("Regresi Linear - Prediksi Repost dari Like")

    fitur_col = 'like_count' if 'like_count' in data.columns else None
    target_col = 'repost_count' if 'repost_count' in data.columns else None

    if fitur_col is None or target_col is None:
        st.error("Kolom `like_count` dan `repost_count` dibutuhkan untuk regresi.")
        return

    df = data.dropna(subset=[fitur_col, target_col]).copy()
    df = df[(df[fitur_col].astype(str).str.strip() != '')
            & (df[target_col].astype(str).str.strip() != '')]
    df[fitur_col] = pd.to_numeric(df[fitur_col], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[fitur_col, target_col]).reset_index(drop=True)

    jumlah_awal = len(data)
    jumlah_valid = len(df)
    jumlah_dibuang = jumlah_awal - jumlah_valid
    st.caption(f"Data awal: **{jumlah_awal}** baris | "
               f"Valid: **{jumlah_valid}** baris | "
               f"Dibuang (like/repost tidak valid): **{jumlah_dibuang}** baris")

    if len(df) < 10:
        st.warning("Data terlalu sedikit untuk regresi (minimal 10 baris).")
        return

    korelasi = df[fitur_col].corr(df[target_col])

    X = df[[fitur_col]]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.caption(f"Data valid: **{len(df)}** baris | "
               f"Latih: **{len(X_train)}** | Uji: **{len(X_test)}** | "
               f"Korelasi like vs repost: **{korelasi:.4f}**")

    st.subheader("Evaluasi Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Korelasi (Pearson)", f"{korelasi:.4f}")
    col2.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
    col3.metric("R² (Koefisien Determinasi)", f"{r2:.4f}")

    with st.expander("Detail koefisien model"):
        st.write(f"**Intercept**: {model.intercept_:.2f}")
        st.write(f"**Koefisien like_count**: {model.coef_[0]:.4f}")

    st.divider()

    st.subheader("Estimasi Repost")

    col1, col2 = st.columns([1, 1], vertical_alignment="bottom")
    likes = col1.number_input("Jumlah like", min_value=0, value=100, step=1)
    tombol = col2.button("Prediksi", width='stretch')

    if tombol:
        input_data = pd.DataFrame({fitur_col: [likes]})
        pred = model.predict(input_data)[0]
        if pred < 0:
            st.warning(f"Estimasi: **{pred:.2f}** - nilai negatif tidak bermakna "
                       f"sebagai jumlah repost (keterbatasan model).")
        else:
            st.success(f"Estimasi repost untuk **{likes} like**: **{pred:.2f} repost**")

    st.divider()

    with st.expander("Repost Aktual vs Prediksi", expanded=False):
        st.caption("Klik icon fullscreen di kanan atas grafik untuk memperbesar")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=20)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', label='Garis ideal')
        ax.set_xlabel('Repost Aktual', fontsize=9)
        ax.set_ylabel('Repost Prediksi', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Exploratory Data Analysis", expanded=False):

        st.markdown("**1. Distribusi Data**")
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
        sns.histplot(df[fitur_col], bins=20, kde=True, ax=axes[0], color='steelblue')
        axes[0].set_title('Like', fontsize=9)
        sns.histplot(df[target_col], bins=20, kde=True, ax=axes[1], color='green')
        axes[1].set_title('Repost', fontsize=9)
        for ax in axes:
            ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**2. Matriks Korelasi**")
        corr = df[[fitur_col, target_col]].corr()
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
            st.dataframe(corr.style.format("{:.3f}"), width='stretch')

        st.markdown("**3. Grafik Regresi (Like vs Repost)**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.regplot(data=df, x=fitur_col, y=target_col, ax=ax,
                    scatter_kws={'alpha': 0.4, 's': 10},
                    line_kws={'color': 'red'})
        ax.set_title('Like vs Repost', fontsize=9)
        ax.set_xlabel('Jumlah Like', fontsize=9)
        ax.set_ylabel('Jumlah Repost', fontsize=9)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    st.subheader("Interpretasi")

    if r2 < 0:
        r2_text = (f"Nilai R2 negatif ({r2:.4f}) menunjukkan model **lebih buruk** "
                   f"daripada hanya memprediksi nilai rata-rata repost.")
    else:
        r2_text = f"Model menjelaskan sekitar **{r2*100:.1f}%** variasi jumlah repost."

    st.markdown(f"""
- **Korelasi ({korelasi:.4f})**: hubungan like_count dan repost_count sangat kuat
  dan searah, sehingga like dapat dipakai untuk memperkirakan repost.
- **MAE ({mae:.2f})**: rata-rata kesalahan prediksi sekitar **+/-{mae:.0f} repost**.
- **R2 ({r2:.4f})**: {r2_text}
- **Koefisien ({model.coef_[0]:.4f})**: setiap tambahan 1 like berasosiasi dengan
  sekitar **{model.coef_[0]:.2f} repost** (intercept {model.intercept_:.2f}).
- **Keterbatasan**: distribusi like dan repost sangat miring, mayoritas postingan
  bernilai 0 dan hanya sedikit yang viral. Korelasi tinggi sebagian besar berasal
  dari segelintir postingan populer, sehingga prediksi untuk nilai kecil kurang stabil.
""")
