from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from waitress import serve
import gdown

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'

# Pastikan folder upload ada, jika tidak buat folder
if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Unduh model dari Google Drive jika belum ada
model_path = "model_age_cnn.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1AQXvpVafi0NMUGD09tXo65lm811fRmYB"  # Ganti dengan link Google Drive Anda
    print("🔽 Mengunduh model dari Google Drive...")
    gdown.download(url, model_path, quiet=False)
    print("✅ Model berhasil diunduh!")

# Fungsi Custom Loss agar bisa dikenali saat Load Model
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Load Model CNN
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={"custom_mse": custom_mse, "MeanSquaredError": tf.keras.losses.MeanSquaredError})
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    print("✅ Model berhasil dimuat dan dikompilasi ulang.")
else:
    raise FileNotFoundError(f"❌ Model tidak ditemukan di {model_path}.")

# Fungsi Preprocessing Gambar
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html', title="Tentang Aplikasi", description="""Aplikasi Identifikasi Umur adalah aplikasi berbasis web yang memanfaatkan teknologi Convolutional Neural Network (CNN) untuk melakukan prediksi umur seseorang berdasarkan gambar wajah mereka.""")

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess dan Prediksi
        img = preprocess_image(filepath)
        prediction = model.predict(img)
        predicted_age = int(prediction[0][0])
        
        return render_template('result.html', age=predicted_age, image=filepath)
    return render_template('index.html')


if __name__ == '__main__':
    print("🚀 Server berjalan di http://127.0.0.1:8080")
    serve(app, host="0.0.0.0", port=8080)
