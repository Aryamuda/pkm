import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
import warnings

# Mengatur agar warning tidak ditampilkan di aplikasi Streamlit (opsional)
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")


# --- Fungsi Generasi Data Dummy ---
# (Sama seperti yang Anda berikan sebelumnya, dengan base_prob = 0.4 dan logika kondisi yang telah disesuaikan)
def combined_resistance_logic_balanced(row, opsi_r_tr_param):
    age = row['Umur']
    sex = row['Jenis Kelamin']
    freq = row['Frekuensi']
    duration = row['Durasi']
    base_prob = 0.4
    age_sex_prob = base_prob
    if (age >= 19 and age <= 29):
        age_sex_prob += 0.1
    if (age >= 50 and age <= 70):
        age_sex_prob += 0.15
    if sex == 'Laki-Laki':
        age_sex_prob += 0.05
    if freq == 'TDS(3x1 Hari)' and duration < 3:
        age_sex_prob += 0.1
    if freq == 'TDS(3x1 Hari)' and duration > 4:
        age_sex_prob += 0.1
    if freq == 'BOD(2x1 hari)' and duration > 5:
        age_sex_prob += 0.15
    if freq == 'OD(1x1 Hari)' and duration > 10:
        age_sex_prob += 0.2
    final_prob = max(0, min(1, age_sex_prob))
    return np.random.choice(opsi_r_tr_param, p=[final_prob, 1 - final_prob])


@st.cache_data  # Cache data generation
def generate_data(jumlah_pasien_param):
    np.random.seed(142)  # Seed untuk reproduktifitas data dummy
    opsi_jenis_kelamin = ['Perempuan', 'Laki-Laki']
    opsi_diagnosa = ['Diabetes mellitus type 2', 'Iskemia jantung', 'PPOK(penyakit paru obstruktif kronik)',
                     'CKD(gagal ginjal kronis)']
    opsi_nama_obat = ['Amoxicillin', 'Ciprofloxacin', 'Azithromycin', 'Ceftriaxone']
    opsi_rute = ['Oral', 'Intravena (suntikan)', 'Intramuskular (infus)']
    opsi_frekuensi = ['BOD(2x1 hari)', 'OD(1x1 Hari)', 'TDS(3x1 Hari)']
    opsi_r_tr = ['Resisten', 'Tidak Resisten']

    data_dict = {
        'No': [f'P{str(i).zfill(4)}' for i in range(1, jumlah_pasien_param + 1)],
        'Umur': np.random.randint(18, 70, jumlah_pasien_param),
        'Jenis Kelamin': np.random.choice(opsi_jenis_kelamin, jumlah_pasien_param, p=[0.5, 0.5]),
        'Diagnosa': np.random.choice(opsi_diagnosa, jumlah_pasien_param),
        'Nama Obat': np.random.choice(opsi_nama_obat, jumlah_pasien_param),
        'Dosis (mg)': np.random.choice([250, 500, 750, 1000], jumlah_pasien_param),
        'Rute': np.random.choice(opsi_rute, jumlah_pasien_param),
        'Frekuensi': np.random.choice(opsi_frekuensi, jumlah_pasien_param),
        'Durasi': np.random.randint(2, 15, jumlah_pasien_param)
    }
    df_generated = pd.DataFrame(data_dict)
    df_generated['R/TR'] = df_generated.apply(lambda row: combined_resistance_logic_balanced(row, opsi_r_tr), axis=1)
    return df_generated


# --- Fungsi Pelatihan Model dan Evaluasi ---
@st.cache_resource  # Cache model terlatih dan hasilnya
def train_and_evaluate_model(_df_input):
    # Persiapan Data (sesuai skrip Anda)
    X = _df_input.drop(['No', 'R/TR'], axis=1)
    y = _df_input['R/TR']

    class_counts = y.value_counts()
    smote_k_neighbors_init = 3  # Sesuai skrip Anda
    min_samples = class_counts.min()

    use_smote = True
    if min_samples <= smote_k_neighbors_init:
        smote_k_neighbors = max(1, min_samples - 1)
        if smote_k_neighbors == 0:  # SMOTE k_neighbors tidak boleh 0
            st.warning(
                f"Tidak dapat menggunakan SMOTE karena kelas minoritas hanya memiliki {min_samples} sampel (k_neighbors akan menjadi 0). SMOTE dilewati.")
            use_smote = False
        else:
            st.info(
                f"Menyesuaikan k_neighbors untuk SMOTE menjadi: {smote_k_neighbors} karena kelas minoritas memiliki {min_samples} sampel.")
    else:
        smote_k_neighbors = smote_k_neighbors_init

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_features = ['Jenis Kelamin', 'Diagnosa', 'Nama Obat', 'Rute', 'Frekuensi']
    numerical_features = ['Umur', 'Dosis (mg)', 'Durasi']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    xgb_params = {
        'alpha': 1.0, 'colsample_bytree': 0.6, 'reg_lambda': 10.0,
        'learning_rate': 0.011624063916904872, 'max_depth': 3,
        'n_estimators': 200, 'subsample': 0.6, 'random_state': 42,
        'eval_metric': 'logloss', 'use_label_encoder': False
    }

    pipeline_steps = [('preprocessor', preprocessor)]
    if use_smote:
        pipeline_steps.append(
            ('sampler', SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=smote_k_neighbors)))
    pipeline_steps.append(('classifier', XGBClassifier(**xgb_params)))

    pipeline_xgb_fixed_params = ImbPipeline(steps=pipeline_steps)

    stratify_param = None
    if len(np.unique(y_encoded)) >= 2:
        stratify_param = y_encoded
    else:
        st.warning("Peringatan: Stratifikasi tidak dapat dilakukan karena hanya ada satu kelas dalam y_encoded.")

    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param
    )

    pipeline_xgb_fixed_params.fit(X_train, y_train_encoded)

    # Feature Importances
    fig_fi = None
    sorted_aggregated_importance_df = None  # Untuk tabel feature importance
    try:
        xgb_model_final = pipeline_xgb_fixed_params.named_steps['classifier']
        preprocessor_final = pipeline_xgb_fixed_params.named_steps['preprocessor']
        feature_names_transformed = preprocessor_final.get_feature_names_out()
        importances = xgb_model_final.feature_importances_
        aggregated_importances = {}

        for feature_name_transformed, importance_score in zip(feature_names_transformed, importances):
            original_feature_name = None
            if feature_name_transformed.startswith('num__'):
                original_feature_name = feature_name_transformed.split('__')[1]
            elif feature_name_transformed.startswith('cat__'):
                transformed_part = feature_name_transformed.split('__')[1]
                for cat_feature in categorical_features:
                    if transformed_part.startswith(cat_feature + "_"):
                        original_feature_name = cat_feature
                        break
                if original_feature_name is None and transformed_part in categorical_features:  # Fallback
                    original_feature_name = transformed_part

            if original_feature_name:
                aggregated_importances.setdefault(original_feature_name, 0.0)
                aggregated_importances[original_feature_name] += importance_score
            else:
                aggregated_importances.setdefault(feature_name_transformed, 0.0)
                aggregated_importances[feature_name_transformed] += importance_score

        if aggregated_importances:
            aggregated_importance_series = pd.Series(aggregated_importances)
            sorted_aggregated_importance = aggregated_importance_series.sort_values(ascending=True)

            # Untuk tabel
            sorted_aggregated_importance_df = sorted_aggregated_importance.reset_index()
            sorted_aggregated_importance_df.columns = ['Fitur', 'Tingkat Kepentingan Relatif']
            sorted_aggregated_importance_df = sorted_aggregated_importance_df.sort_values(
                by='Tingkat Kepentingan Relatif', ascending=False)

            fig_fi, ax_fi = plt.subplots(figsize=(10, max(6, len(sorted_aggregated_importance) * 0.4)))  # Adjusted size
            sorted_aggregated_importance.plot(kind='barh', ax=ax_fi, color='darkcyan')
            ax_fi.set_title(f'Variabel Penting Dalam Penentuan Resistensi')
            ax_fi.set_xlabel('Total Relatif Kepentingan')
            ax_fi.set_ylabel('Variabel')
            plt.tight_layout()  # Ensure everything fits
    except Exception as e:
        st.error(f"Tidak dapat membuat plot feature importance: {e}")

    # Evaluation Metrics
    y_pred_encoded = pipeline_xgb_fixed_params.predict(X_test)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
    y_test_original = label_encoder.inverse_transform(y_test_encoded)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test_original, y_pred_original)

    positive_label_display = "N/A"
    # Menentukan label positif untuk presisi/recall/F1
    if 'Resisten' in label_encoder.classes_:
        positive_label = 'Resisten'
        positive_label_display = positive_label
    elif len(label_encoder.classes_) > 1:
        positive_label = label_encoder.classes_[1]
        positive_label_display = positive_label
        st.warning(
            f"Peringatan: Kelas 'Resisten' tidak ditemukan dalam label target. Menggunakan '{positive_label}' sebagai kelas positif untuk metrik.")
    elif len(label_encoder.classes_) == 1:
        positive_label = label_encoder.classes_[0]
        positive_label_display = positive_label
        st.warning(
            f"Peringatan: Hanya ada satu kelas ('{positive_label}') dalam data. Metrik Precision/Recall/F1 mungkin tidak valid atau bermakna.")
    else:
        positive_label = None
        st.error("Kesalahan: Tidak ada kelas yang terdeteksi oleh LabelEncoder. Tidak dapat menghitung metrik.")

    metrics['positive_label_for_display'] = positive_label_display
    if positive_label:
        metrics['precision'] = precision_score(y_test_original, y_pred_original, pos_label=positive_label,
                                               zero_division=0)
        metrics['recall'] = recall_score(y_test_original, y_pred_original, pos_label=positive_label, zero_division=0)
        metrics['f1_score'] = f1_score(y_test_original, y_pred_original, pos_label=positive_label, zero_division=0)
    else:  # Default jika positive_label tidak bisa ditentukan
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1_score'] = 0.0

    report_dict = None
    if len(label_encoder.classes_) > 0:
        target_names_report = [str(cls) for cls in label_encoder.classes_]
        try:
            report_dict = classification_report(y_test_original, y_pred_original, target_names=target_names_report,
                                                zero_division=0, output_dict=True)
        except ValueError as ve:  # Handle case where y_true or y_pred is empty or has types that cannot be processed
            st.error(f"Gagal membuat classification report: {ve}")

    # Confusion Matrix
    fig_cm = None
    if len(label_encoder.classes_) > 0:
        cm_xgb = confusion_matrix(y_test_original, y_pred_original, labels=label_encoder.classes_)
        fig_cm, ax_cm = plt.subplots(figsize=(7, 5))  # Adjusted size
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='BuPu',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        plt.tight_layout()

    return fig_fi, sorted_aggregated_importance_df, metrics, report_dict, fig_cm, list(label_encoder.classes_)


# --- UI Streamlit ---
st.title("Analisis & Prediksi Resistensi Antibiotik (Simulasi)")
st.markdown("""
Selamat datang di aplikasi simulasi untuk analisis resistensi antibiotik menggunakan model XGBoost.
Aplikasi ini memungkinkan Anda untuk:
1.  Menghasilkan dataset pasien sintetis dengan parameter yang dapat diatur.
2.  Melatih model _machine learning_ untuk memprediksi status resistensi ('Resisten' atau 'Tidak Resisten').
3.  Melihat faktor-faktor apa saja yang paling mempengaruhi prediksi model.
4.  Mengevaluasi performa model melalui berbagai metrik.

Ini adalah alat untuk demonstrasi konsep dan eksplorasi.
""")

st.sidebar.header("Pengaturan Dataset Sintetis")
jumlah_pasien_input = st.sidebar.number_input("Jumlah Pasien", min_value=100, max_value=10000, value=1000, step=100,
                                              help="Atur jumlah data pasien yang akan digenerasi untuk simulasi.")

# Generate data
df = generate_data(jumlah_pasien_input)

st.subheader("Pratinjau Dataset Sintetis")
st.markdown(f"Dataset yang dihasilkan memiliki **{df.shape[0]}** baris dan **{df.shape[1]}** kolom.")

col1_info, col2_info = st.columns([0.7, 0.3])  # Atur lebar kolom
with col1_info:
    st.write("Lima baris pertama data:")
    st.dataframe(df.head())
with col2_info:
    st.write("Distribusi Target 'R/TR':")
    distribusi_target = df['R/TR'].value_counts().reset_index()
    distribusi_target.columns = ['Kelas Target', 'Jumlah']
    st.dataframe(distribusi_target)

if st.sidebar.button("Latih Model & Tampilkan Hasil Analisis", type="primary", use_container_width=True):
    if df['R/TR'].nunique() < 2:
        st.error("Kesalahan: Data yang dihasilkan hanya memiliki satu kelas pada kolom target 'R/TR'. "
                 "Model klasifikasi tidak dapat dilatih. Coba variasikan jumlah pasien atau periksa kembali logika generasi data.")
    else:
        with st.spinner(f"Sedang memproses {jumlah_pasien_input} data dan melatih model XGBoost... Mohon tunggu."):
            fig_fi, fi_df, metrics, report_dict, fig_cm, le_classes = train_and_evaluate_model(df.copy())

        st.success("Analisis selesai! Berikut adalah hasilnya:")

        tab_metrics, tab_fi, tab_report = st.tabs(
            ["Metrik & Confusion Matrix", "Pentingnya Fitur", "Laporan Klasifikasi Rinci"])

        with tab_metrics:
            st.header("Evaluasi Performa Model")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            m_col2.metric(f"Precision ({metrics['positive_label_for_display']})", f"{metrics['precision']:.4f}")
            m_col3.metric(f"Recall ({metrics['positive_label_for_display']})", f"{metrics['recall']:.4f}")
            m_col4.metric(f"F1-Score ({metrics['positive_label_for_display']})", f"{metrics['f1_score']:.4f}")

            st.subheader("Confusion Matrix")
            if fig_cm:
                st.pyplot(fig_cm, use_container_width=False)  # Agar tidak terlalu lebar
            else:
                st.info(
                    "Tidak dapat menampilkan Confusion Matrix (kemungkinan karena hanya satu kelas terdeteksi atau masalah lain).")

        with tab_fi:
            st.header("Pentingnya Fitur (Feature Importance)")
            st.markdown("""
            Grafik dan tabel di bawah ini menunjukkan fitur-fitur mana yang dianggap paling penting oleh model XGBoost 
            dalam membuat prediksi status resistensi. Fitur dengan nilai kepentingan relatif yang lebih tinggi memiliki pengaruh yang lebih besar.
            """)
            if fig_fi:
                st.pyplot(fig_fi, use_container_width=True)  # Agar plot bisa responsif
            else:
                st.info("Tidak dapat menampilkan grafik Feature Importance.")

            if fi_df is not None:
                st.write("Data Tingkat Kepentingan Fitur:")
                st.dataframe(fi_df)
            else:
                st.info("Tidak ada data feature importance untuk ditampilkan dalam tabel.")

        with tab_report:
            st.header("Laporan Klasifikasi Rinci")
            if report_dict:
                report_df = pd.DataFrame(report_dict).transpose()
                # Format support sebagai integer jika ada
                if 'support' in report_df.columns:
                    try:  # Pastikan konversi aman
                        report_df['support'] = report_df['support'].apply(lambda x: int(x) if pd.notnull(x) else x)
                    except ValueError:
                        pass  # Biarkan jika ada nilai non-numerik yang tidak bisa dikonversi
                st.dataframe(report_df)
            else:
                st.info(
                    "Tidak dapat menampilkan Laporan Klasifikasi (kemungkinan karena hanya satu kelas terdeteksi atau masalah lain).")

        st.markdown("---")
        st.caption(
            f"Model dilatih dan dievaluasi pada dataset yang baru digenerasi dengan {jumlah_pasien_input} pasien")

else:
    st.info("Atur jumlah pasien di sidebar dan klik tombol 'Latih Model & Tampilkan Hasil Analisis' untuk memulai.")

st.sidebar.markdown("---")
st.sidebar.markdown("Aplikasi Streamlit untuk PKM-GFT")
st.sidebar.markdown("Dibuat sebagai contoh")
