import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("rice_model.keras")

# Load label
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

st.title("🌾 Deteksi Penyakit Daun Padi")

option = st.radio("Pilih metode:", ["Upload", "Kamera"])

image = None

if option == "Upload":
    file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file)

if option == "Kamera":
    cam = st.camera_input("Ambil foto")
    if cam:
        image = Image.open(cam)

if image:
    st.image(image)

    # preprocessing
    img = image.resize((224,224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[:,:,:3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # prediksi
    pred = model.predict(img)
    index = np.argmax(pred)
    conf = np.max(pred)

    st.success(f"Hasil: {labels[index]}")
    st.info(f"Akurasi: {conf*100:.2f}%")
