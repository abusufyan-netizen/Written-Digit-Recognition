# model_utils.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import streamlit as st

MODEL_PATH = "mnist_cnn.keras"

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    # If model exists, load it.
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.warning(f"Failed loading model: {e}. Retraining.")
            os.remove(MODEL_PATH)

    # Train a compact CNN (quick) and save
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1,28,28,1)
    x_test  = x_test.reshape(-1,28,28,1)

    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train for a small number of epochs (fast). Increase epochs for better accuracy.
    with st.spinner("Training model (fast mode)..."):
        model.fit(x_train, y_train, epochs=10, validation_split=0.1, verbose=2)

    model.save(MODEL_PATH)
    return model
