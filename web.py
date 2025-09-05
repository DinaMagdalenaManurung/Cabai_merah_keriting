import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os
import random
import matplotlib.dates as mdates
from PIL import Image

# ==============================
# Custom CSS untuk latar belakang
# ==============================
page_bg = """
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffb6c1; /* pink tua */
    color: black;
}

/* Halaman utama */
.stApp {
    background-color: rgba(255, 182, 193, 0.3); /* pink muda transparan */
    color: black;
}

/* Kotak info (Tentang Model) */
.stAlert {
    background-color: #fffaf0 !important; /* cream */
    color: black !important;
    border-radius: 10px;
}

/* Judul */
h1, h2, h3, h4 {
    color: #8B0000; /* merah tua */
}

/* Tombol */
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 14px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ==============================
# Seed global untuk deterministik
# ==============================
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism()
tf.config.set_visible_devices([], 'GPU')  # optional: paksa CPU

# ==============================
# Cache: Load Models & Scalers
# ==============================
@st.cache_resource
def load_models():
    arimax_model = joblib.load("arimax_model_full.pkl")
    ann_model = tf.keras.models.load_model("ann_model.keras")
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return arimax_model, ann_model, scaler_X, scaler_y

# ==============================
# Helpers: Session State
# ==============================
if "step" not in st.session_state:
    st.session_state.step = "upload_hist"
if "hist_data" not in st.session_state:
    st.session_state.hist_data = None
if "exog_data" not in st.session_state:
    st.session_state.exog_data = None
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None
if "predict_days" not in st.session_state:
    st.session_state.predict_days = 5

def go_to(step_name: str):
    st.session_state.step = step_name

# ==============================
# Sidebar
# ==============================
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigasi", ["Home", "Prediksi Harga"], key="menu_radio")

st.sidebar.markdown("---")
st.sidebar.subheader("Tentang Model")

st.sidebar.markdown(
    """
    <div style="background-color:#fffaf0;
                padding:10px;
                border-radius:10px;
                border:1px solid #f5deb3;
                color:black;">
    <b>Model Hybrid ARIMAX–ANN:</b><br>
    • <b>ARIMAX(3,0,0)</b> memodelkan tren utama harga dengan variabel eksogen.<br>
    • <b>ANN</b> memodelkan <i>residual</i> ARIMAX (1 layer, input_neuron=5, hidden_neuron=32).<br>
    • <b>Hybrid</b> = ARIMAX + Residual ANN.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================
# ===============  HOME  ==================
# =========================================
if menu == "Home":
    st.title("Informasi Cabai Merah Keriting 🌶️")

    # Gambar dengan ukuran diatur
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        try:
            img = Image.open("gambar_cabai.jpg")
            st.image(img, caption="Cabai Merah Keriting", width=400)  # atur lebar (tinggi ikut proporsi)
        except Exception:
            st.markdown("🌶️ *(Gambar tidak dapat dimuat — periksa path file)*")

    st.markdown(
        "Cabai merah keriting adalah komoditas penting di Indonesia. Cabai merah keriting sering digunakan sebagai bumbu masakan dan juga dipakai sebagai bahan baku produk kesehatan. "
        "Permintaan terhadap cabai merah keriting sangat meningkat, sehingga harganya sering mengalami kenaikan berkali-kali lipat. "
        "Harga cabai merah keriting di kota Medan sangat dipengaruhi oleh pasokan dari luar daerah, curah hujan, dan pekan hari besar/keagamaan."
    )

    st.markdown("### Visualisasi Harga Cabai Merah Keriting di Kota Medan (1 Januari 2019 - 28 Februari 2025)")
    data_home = None
    if st.session_state.hist_data is not None:
        data_home = st.session_state.hist_data.copy()
    elif os.path.exists("data_harga_cabai.csv"):
        try:
            df_tmp = pd.read_csv("data_harga_cabai.csv")
            df_tmp["Tanggal"] = pd.to_datetime(df_tmp["Tanggal"])
            data_home = df_tmp
        except Exception:
            data_home = None

    if data_home is None:
        up = st.file_uploader("Upload data untuk ditampilkan di Home", type="csv", key="home_uploader")
        if up:
            data_home = pd.read_csv(up)
            data_home["Tanggal"] = pd.to_datetime(data_home["Tanggal"])

    if data_home is not None and len(data_home):
        data_home = data_home.sort_values("Tanggal").reset_index(drop=True)
        min_d, max_d = data_home["Tanggal"].min().date(), data_home["Tanggal"].max().date()
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Tanggal awal", min_d, min_value=min_d, max_value=max_d, key="home_start")
        with c2:
            end_date = st.date_input("Tanggal akhir", max_d, min_value=min_d, max_value=max_d, key="home_end")

        mask = (data_home["Tanggal"] >= pd.to_datetime(start_date)) & (data_home["Tanggal"] <= pd.to_datetime(end_date))
        df_filtered = data_home.loc[mask]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_filtered["Tanggal"], df_filtered["Harga Cabai"], label="Harga Cabai", color ='red')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (Rp)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Belum ada data untuk divisualisasikan. Silakan upload di sini atau di menu **Prediksi Harga**.")

# =========================================
# ============  PREDIKSI HARGA  ===========
# =========================================
else:
    st.title("🌶️Prediksi Harga Cabai Merah Keriting (Hybrid ARIMAX–ANN)🌶️")

    # ---------- STEP 1: Upload & Preprocess ----------
    st.subheader("1) Upload Data Historis")
    st.caption("Contoh struktur data (perhatikan nama kolom dan tipe data):")
    contoh_data = pd.DataFrame({
        "Tanggal": pd.date_range("2019-01-01", periods=2, freq="D"),
        "Harga Cabai": [20000, 21000],
        "Curah Hujan": [1.55, 2.50],
        "Pekan Hari Besar": [True, False],
    })
    st.dataframe(contoh_data)

    uploaded_file = st.file_uploader("Upload CSV Data Historis", type="csv", key="hist_uploader")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = data.dropna()
        expected_cols = ["Tanggal", "Harga Cabai", "Curah Hujan", "Pekan Hari Besar"]
        if not all(col in data.columns for col in expected_cols):
            st.error(f"Kolom harus mengandung: {expected_cols}")
        else:
            data["Harga Cabai"] = pd.to_numeric(data["Harga Cabai"], errors='coerce')
            data["Curah Hujan"] = pd.to_numeric(data["Curah Hujan"], errors='coerce')
            if data["Pekan Hari Besar"].dtype == object:
                data["Pekan Hari Besar"] = data["Pekan Hari Besar"].map({"True": 1, "False": 0})
            data = data.dropna().reset_index(drop=True)

            # Outlier (IQR -> Winsorize P5/P95)
            q1 = data["Harga Cabai"].quantile(0.25)
            q3 = data["Harga Cabai"].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            p5 = data["Harga Cabai"].quantile(0.05)
            p95 = data["Harga Cabai"].quantile(0.95)
            data["Harga Cabai"] = data["Harga Cabai"].apply(lambda x: p5 if x < lower else (p95 if x > upper else x))

            data["Tanggal"] = pd.to_datetime(data["Tanggal"])
            data = data.sort_values("Tanggal").reset_index(drop=True)

            st.session_state.hist_data = data.copy()

            st.markdown("#### Preview Data Historis")
            st.dataframe(data.head())

            st.markdown("#### Statistik Deskriptif")
            st.write(data.describe())

            st.markdown("#### Visualisasi Data Historis")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(data["Tanggal"], data["Harga Cabai"], label="Harga Cabai", color ='red')
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga Cabai (Rp)")
            ax.legend()
            st.pyplot(fig)

            st.button("Prediksi Masa Depan", key="btn_next_to_predict",
                      on_click=lambda: go_to("predict_params"))

    # ---------- STEP 2: Parameter & Eksogen ----------
    if st.session_state.step == "predict_params" and st.session_state.hist_data is not None:
        st.markdown("---")
        st.subheader("2) Tentukan Panjang Prediksi & Upload Variabel Eksogen")
        st.session_state.predict_days = st.number_input(
            "Berapa hari yang ingin diprediksi?",
            min_value=1, value=st.session_state.predict_days, step=1, key="input_days"
        )

        st.caption("Contoh struktur variabel eksogen :")
        contoh_exog = pd.DataFrame({
            "Curah Hujan": [7.21, 1.66],
            "Pekan Hari Besar": [False, True],
        })
        st.dataframe(contoh_exog)

        exog_file = st.file_uploader(
            f"Upload CSV Variabel Eksogen untuk {st.session_state.predict_days} hari ke depan",
            type="csv", key="exog_uploader"
        )

        ok_to_predict = False
        exog_future = None
        if exog_file is not None:
            exog_future = pd.read_csv(exog_file)
            required_cols = ["Curah Hujan", "Pekan Hari Besar"]
            if not all(col in exog_future.columns for col in required_cols):
                st.error(f"Kolom eksogen harus mengandung: {required_cols}")
            elif len(exog_future) != st.session_state.predict_days:
                st.error("❌ Banyaknya data eksogen harus sesuai dengan jumlah hari yang ingin diprediksi.")
            else:
                exog_future = exog_future[required_cols]
                exog_future["Curah Hujan"] = pd.to_numeric(exog_future["Curah Hujan"], errors='coerce')
                if exog_future["Pekan Hari Besar"].dtype == object:
                    exog_future["Pekan Hari Besar"] = exog_future["Pekan Hari Besar"].map({"True": 1, "False": 0})
                exog_future = exog_future.dropna().astype(float).reset_index(drop=True)
                st.session_state.exog_data = exog_future.copy()
                ok_to_predict = True

        colA, colB, colC = st.columns([3,2,3])
        with colB:
            st.button("Prediksi", key="btn_predict",
                      disabled=not ok_to_predict,
                      on_click=lambda: go_to("show_result"))

# ---------- STEP 3: Hasil ----------
if st.session_state.step == "show_result":
    data = st.session_state.hist_data
    exog_future = st.session_state.exog_data
    predict_days = st.session_state.predict_days

    if data is None or exog_future is None:
        st.warning("Lengkapi langkah sebelumnya terlebih dahulu.")
    else:
        arimax_model, ann_model, scaler_X, scaler_y = load_models()

        y_arimax_vals = arimax_model.forecast(steps=predict_days, exog=exog_future)
        y_arimax_5 = pd.Series(np.asarray(y_arimax_vals).ravel(), name="Prediksi_ARIMAX")

        residual_hist = pd.Series(arimax_model.resid).dropna()
        n_lag = 5
        seed_last = residual_hist.values[-n_lag:]

        def forecast_residual_ann_recursive(model_ann, scaler_X, scaler_y, last_resid_vec, steps, n_lag):
            hist = list(last_resid_vec.astype(float))
            preds = []
            for _ in range(steps):
                x = np.array(hist[-n_lag:]).reshape(1, -1)
                x_scaled = scaler_X.transform(x)
                y_scaled = model_ann.predict(x_scaled, verbose=0)
                y_pred = scaler_y.inverse_transform(y_scaled)[0, 0]
                preds.append(y_pred)
                hist.append(y_pred)
            return np.array(preds)

        resid_pred_5_vals = forecast_residual_ann_recursive(
            ann_model, scaler_X, scaler_y, seed_last, predict_days, n_lag
        )
        y_resid_ann_5 = pd.Series(resid_pred_5_vals, name="Prediksi_Residual_ANN")

        y_hybrid_5 = (y_arimax_5 + y_resid_ann_5).rename("Prediksi_Hybrid")

        future_dates = pd.date_range(start=data["Tanggal"].iloc[-1] + pd.Timedelta(days=1),
                                     periods=predict_days)
        hasil_pred = pd.DataFrame({"Tanggal": future_dates})
        if isinstance(exog_future, pd.DataFrame):
            hasil_pred = pd.concat([hasil_pred, exog_future.reset_index(drop=True)], axis=1)
        hasil_pred["Prediksi_Hybrid"] = y_hybrid_5.values

        st.session_state.pred_result = hasil_pred.copy()

        st.markdown("### Hasil Prediksi Hybrid")
        st.dataframe(
            hasil_pred.style.format(precision=2, thousands=","),
            use_container_width=True
        )

        st.markdown("### Visualisasi Prediksi Hybrid")
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(
            hasil_pred["Tanggal"],
            hasil_pred["Prediksi_Hybrid"],
            label="Prediksi Hybrid",
            marker='o',
            color='red'
        )
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Harga Cabai")
        ax2.legend()

        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
        fig2.autofmt_xdate(rotation=45)

        st.pyplot(fig2)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.button("Ubah Parameter", key="btn_back_params", on_click=lambda: go_to("predict_params"))
        c2.button("Mulai Ulang", key="btn_restart", on_click=lambda: go_to("upload_hist"))