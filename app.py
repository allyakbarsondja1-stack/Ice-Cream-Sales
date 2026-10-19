"""Aplikasi Flask untuk prediksi pendapatan es krim berdasarkan suhu udara."""

import json
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "ice_cream_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model belum dibuat. Jalankan train_model.py terlebih dahulu.")

with MODEL_PATH.open("rb") as model_file:
    model = joblib.load(model_file)

with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)


def predict_revenue(temperature):
    """Mengeksekusi prediksi berdasarkan model linear sederhana."""
    temperature_value = float(temperature)
    prediction = model.predict([[temperature_value]])[0]
    return round(float(prediction), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    """Menampilkan halaman utama dan menangani input prediksi."""
    temperature_value = ""
    prediction = None
    error_message = ""
    warning_message = ""

    if request.method == "POST":
        temperature_value = request.form.get("temperature", "").strip()

        try:
            temperature = float(temperature_value)

            if not 0 <= temperature <= 45:
                warning_message = (
                    "Suhu berada di luar rentang data latih (0-45°C), "
                    "sehingga prediksi bersifat ekstrapolasi dan perlu interpretasi hati-hati."
                )

            prediction = predict_revenue(temperature)
        except (TypeError, ValueError):
            error_message = "Input suhu tidak valid. Masukkan angka suhu dalam satuan °C."
        except Exception:
            error_message = "Prediksi gagal diproses. Silakan coba lagi."

    return render_template(
        "index.html",
        temperature_value=temperature_value,
        prediction=prediction,
        metrics=metrics,
        error_message=error_message,
        warning_message=warning_message,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON untuk prediksi suhu ke pendapatan."""
    payload = request.get_json(silent=True) or {}
    temperature = payload.get("temperature")

    if temperature is None:
        return jsonify({"error": "Parameter temperature harus ada."}), 400

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return jsonify({"error": "Temperature harus berupa angka."}), 400

    revenue = predict_revenue(temperature)
    return jsonify({"revenue": revenue})


@app.route("/health")
def health_check():
    """Endpoint kesehatan untuk cek status aplikasi."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
