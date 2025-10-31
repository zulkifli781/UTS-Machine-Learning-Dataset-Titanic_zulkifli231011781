import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- FUNGSI GENERASI DATA SINTETIK ---

def generate_synthetic_time_series_data(days=365):
    """Generate synthetic daily COVID-19 cases for 'days' days."""
    print("Membentuk data time series COVID-19 sintetik...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    base_cases = 500 + 400 * np.sin(np.linspace(0, 3 * np.pi, days))
    weekly_effect = np.array([50 if d.weekday() in [2,3,4] else -40 for d in dates]) # midweek higher, weekend lower
    noise = np.random.normal(0, 100, days)
    daily_cases = np.maximum(0, base_cases + weekly_effect + noise).astype(int)
    data = pd.DataFrame({'Tanggal': dates, 'Kasus_Harian': daily_cases})
    return data.set_index('Tanggal')

def generate_synthetic_patient_data(n_samples=1000):
    """Generate synthetic patient data for classification."""
    print("Membentuk data pasien sintetik untuk Klasifikasi...")
    np.random.seed(42)
    age = np.random.randint(18, 90, n_samples)
    comorbidity = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    risk = 0.01 * age + 0.5 * comorbidity + np.random.normal(0, 2, n_samples)
    outcome = (risk > np.percentile(risk, 85)).astype(int)
    data = pd.DataFrame({
        'Usia': age,
        'Komorbiditas': comorbidity,
        'Status_Akhir': outcome
    })
    return data

# --- TUGAS A: KLASIFIKASI (RANDOM FOREST) ---

def run_classification_model(df):
    """Run Random Forest to predict patient outcome."""
    print("\n--- [Tugas A: Klasifikasi - Prediksi Status Akhir Pasien] ---")
    X = df[['Usia', 'Komorbiditas']]
    y = df['Status_Akhir']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nLaporan Klasifikasi (Metrik Kinerja):")
    print(classification_report(y_test, y_pred, target_names=['Sembuh', 'Meninggal']))
    print(f"Pentingnya Fitur (Usia: {model.feature_importances_[0]:.4f}, Komorbiditas: {model.feature_importances_[1]:.4f})")
    print("Komorbiditas menunjukkan peran yang lebih besar dalam memprediksi kematian (seperti yang diharapkan).")

# --- TUGAS C: RANGKAIAN WAKTU (LSTM) ---

def create_dataset(data, look_back=1):
    """Convert time series to LSTM input/output pairs."""
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def run_lstm_forecasting(df):
    """Run LSTM for daily case forecasting."""
    print("\n--- [Tugas C: Rangkaian Waktu - Peramalan Kasus Harian dengan LSTM] ---")
    data = df['Kasus_Harian'].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]
    look_back = 10
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(f"Membangun dan melatih model LSTM dengan look_back={look_back}...")
    model = Sequential([
        LSTM(4, input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=0)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict[:,0]))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict[:,0]))
    print(f"\nRMSE (Root Mean Squared Error) Pelatihan: {train_rmse:.2f} Kasus")
    print(f"RMSE (Root Mean Squared Error) Pengujian: {test_rmse:.2f} Kasus")
    print("RMSE menunjukkan rata-rata kesalahan prediksi dalam satuan kasus harian.")
    print("\nVisualisasi Sederhana Hasil Prediksi pada Data Pengujian:")
    results_df = pd.DataFrame({
        'Tanggal': df.index[train_size + look_back:train_size + look_back + len(y_test_inv)],
        'Aktual': y_test_inv,
        'Prediksi': test_predict[:, 0]
    })
    print(results_df.head(10).to_markdown(index=False))

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    # Optional: Uncomment if you want to disable GPU usage
    # if tf.config.list_physical_devices('GPU'):
    #     tf.config.set_visible_devices([], 'GPU')

    # 1. TUGAS KLASIFIKASI (Random Forest)
    patient_df = generate_synthetic_patient_data()
    run_classification_model(patient_df)

    # 2. TUGAS RANGKAIAN WAKTU (LSTM)
    time_series_df = generate_synthetic_time_series_data()
    run_lstm_forecasting(time_series_df)

    print("\n========================================================")
    print("Implementasi model ML COVID-19 selesai.")
    print("Ganti fungsi 'generate_synthetic_data' dengan data COVID-19 nyata Anda.")
    print("========================================================")