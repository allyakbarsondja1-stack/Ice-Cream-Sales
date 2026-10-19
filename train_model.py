"""Pelatihan model regresi linear sederhana untuk prediksi pendapatan es krim."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "Ice_Cream.csv"
MODEL_DIR = BASE_DIR / "model"
IMG_DIR = BASE_DIR / "static" / "img"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def display_data_overview(df):
    """Menampilkan ringkasan dataset sesuai tugas EDA."""
    print("\n=== HEAD ===")
    print(df.head().to_string(index=False))

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== INFO ===")
    print(df.info())

    print("\n=== DESCRIBE ===")
    print(df.describe().to_string())


def check_missing_and_duplicates(df):
    """Mengecek missing values dan duplikat."""
    missing_values = df.isna().sum()
    duplicated_rows = df.duplicated().sum()

    print("\n=== MISSING VALUES ===")
    print(missing_values.to_string())

    print("\n=== DUPLICATE ROWS ===")
    print(f"Jumlah baris duplikat: {duplicated_rows}")

    if duplicated_rows > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Baris duplikat berhasil dihapus. Dataset menjadi {len(df)} baris.")
    return df


def detect_outliers(df):
    """Mendeteksi outlier menggunakan metode IQR untuk dua kolom numerik."""
    print("\n=== OUTLIER DETECTION (IQR) ===")
    outlier_summary = {}

    for column in ["Temperature", "Revenue"]:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        outlier_summary[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(outlier_count),
        }

        print(
            f"{column}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
            f"batas bawah={lower_bound:.2f}, batas atas={upper_bound:.2f}, "
            f"jumlah outlier={outlier_count}"
        )

    print(
        "Keputusan: mempertahankan data karena batas valid pada domain bisnis "
        "(Temperature 0-45°C, Revenue 10-1000 USD) masih masuk akal dan tidak "
        "menunjukkan kesalahan input yang jelas."
    )
    return df


def save_scatter_plot(df):
    """Menyimpan scatter plot hubungan suhu dan pendapatan."""
    plt.figure(figsize=(9, 6))
    plt.scatter(df["Temperature"], df["Revenue"], alpha=0.7, color="royalblue")
    plt.title("Scatter Plot: Suhu vs Pendapatan Es Krim")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Revenue (USD)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "scatter_plot.png", dpi=200)
    plt.close()


def build_model_and_evaluate(df):
    """Membuat model regresi linear sederhana dan mengevaluasi performa."""
    X = df[["Temperature"]]
    y = df["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=== MODEL EVALUATION ===")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    intercept = float(model.intercept_)
    coef = float(model.coef_[0])
    print(f"Intercept: {intercept:.4f}")
    print(f"Koefisien Temperature: {coef:.4f}")

    print("\n=== INTERPRETASI MODEL BERDASARKAN STUDI KASUS ===")
    print(
        f"Koefisien sebesar {coef:.2f} USD/°C berarti setiap kenaikan suhu 1°C "
        "diperkirakan meningkatkan pendapatan penjualan es krim sekitar "
        f"{coef:.2f} USD, dengan asumsi faktor lain konstan."
    )
    print(
        f"Intercept sebesar {intercept:.2f} USD mencerminkan prediksi pendapatan saat suhu 0°C. "
        "Nilai ini masih masuk akal secara bisnis untuk menggambarkan tren dasar, "
        "meskipun dalam praktik pendapatan pada suhu 0°C kemungkinan tidak selalu relevan "
        "karena penjualan es krim cenderung turun di cuaca dingin."
    )
    print(
        f"RMSE sekitar {rmse:.2f} USD menunjukkan rata-rata selisih prediksi terhadap data aktual "
        f"sekitar {rmse:.2f} USD, yang relatif kecil dibanding rata-rata pendapatan "
        f"{y.mean():.2f} USD."
    )

    return model, X_train, X_test, y_train, y_test, y_pred, intercept, coef, r2, rmse, mae


def save_regression_plots(model, X_train, X_test, y_train, y_test, y_pred):
    """Menyimpan plot garis regresi, aktual vs prediksi, dan residual plot."""
    x_values = np.linspace(0, 45, 200).reshape(-1, 1)
    predicted_line = model.predict(x_values)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train["Temperature"], y_train, color="tab:blue", alpha=0.7, label="Data Latih")
    plt.scatter(X_test["Temperature"], y_test, color="tab:orange", alpha=0.7, label="Data Uji")
    plt.plot(x_values, predicted_line, color="red", linewidth=2, label="Garis Regresi")
    plt.title("Garis Regresi Linear: Suhu vs Pendapatan")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Revenue (USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "regression_line.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.8, color="green")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5)
    plt.title("Actual vs Predicted Revenue")
    plt.xlabel("Actual Revenue (USD)")
    plt.ylabel("Predicted Revenue (USD)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close()

    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test["Temperature"], residuals, alpha=0.8, color="purple")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Residual Plot")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Residual")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "residual_plot.png", dpi=200)
    plt.close()


def save_metrics(intercept, coef, r2, rmse, mae, n_train, n_test, df):
    """Menyimpan hasil metrik untuk aplikasi web."""
    metrics = {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "intercept": float(intercept),
        "coef": float(coef),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "temp_min": float(df["Temperature"].min()),
        "temp_max": float(df["Temperature"].max()),
        "revenue_mean": float(df["Revenue"].mean()),
    }

    metrics_path = MODEL_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\n=== METRICS DISIMPAN ===\n{json.dumps(metrics, indent=2)}")
    return metrics


def main():
    """Proses utama EDA, pelatihan, dan penyimpanan artefak model."""
    dataset_path = DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di {dataset_path}.")

    df = pd.read_csv(dataset_path)
    display_data_overview(df)

    df = check_missing_and_duplicates(df)
    df = df.astype({"Temperature": float, "Revenue": float})
    df = detect_outliers(df)

    corr = df["Temperature"].corr(df["Revenue"], method="pearson")
    print(f"\n=== PEARSON CORRELATION ===")
    print(f"Koefisien korelasi Pearson: {corr:.4f}")
    print("Interpretasi: hubungan linear positif yang kuat antara suhu dan pendapatan es krim.")

    save_scatter_plot(df)
    print(f"\nScatter plot berhasil disimpan di: {IMG_DIR / 'scatter_plot.png'}")

    model, X_train, X_test, y_train, y_test, y_pred, intercept, coef, r2, rmse, mae = build_model_and_evaluate(df)
    save_regression_plots(model, X_train, X_test, y_train, y_test, y_pred)

    model_path = MODEL_DIR / "ice_cream_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel final berhasil disimpan di: {model_path}")

    metrics = save_metrics(intercept, coef, r2, rmse, mae, len(X_train), len(X_test), df)

    print("\n=== TRAINING SELESAI ===")
    print(f"Total data akhir: {len(df)} baris")
    print(f"Model R²: {metrics['r2']:.4f}")
    print(f"Model RMSE: {metrics['rmse']:.2f}")


if __name__ == "__main__":
    main()
