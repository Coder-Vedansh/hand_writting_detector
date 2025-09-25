# -------------------------------
# Handwritten Digit Recognition App
# -------------------------------

import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# Model Path
# -------------------------------
MODEL_PATH = "mnist_model.h5"
abs_model_path = os.path.abspath(MODEL_PATH)
st.write(f"📂 Model Path: {abs_model_path}")  # Display model path in the app


# -------------------------------
# Train or Load Model (cached)
# -------------------------------
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        # Load dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Preprocess
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = x_train.reshape((-1, 28, 28, 1))
        x_test = x_test.reshape((-1, 28, 28, 1))
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Build model
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
        ])

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test), verbose=1)

        # Save model
        model.save(MODEL_PATH)
    else:
        model = load_model(MODEL_PATH)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


model = load_or_train_model()

# -------------------------------
# Streamlit Web App
# -------------------------------
st.title("✍ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) in the box below, then click **Predict!**")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert RGBA → grayscale 28x28
        img = canvas_result.image_data[:, :, :3]  # drop alpha channel
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Prediction
        pred = model.predict(img)[0]
        digit = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.success(f"✅ Predicted Digit: {digit} (Confidence: {confidence:.2f}%)")
    else:
        st.warning("⚠ Please draw a digit before predicting.")
