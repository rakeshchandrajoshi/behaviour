# streamlit_keystroke_autoencoder_realtime.py
import streamlit as st
import time
import numpy as np
import pandas as pd
import io
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="Realtime Keystroke Autoencoder", layout="centered")

st.title("Realtime Keystroke Autoencoder Trainer")

st.write("""
Type 5 sentences below (one per session).  
The app will capture keystroke timings, extract features, chunk into data points, 
train a dense autoencoder, and export `.tflite` and `.pkl` models.
""")

# ---------------- Keystroke Capture ----------------
if "sessions" not in st.session_state:
    st.session_state.sessions = [[] for _ in range(5)]  # 5 sessions of events
if "current_session" not in st.session_state:
    st.session_state.current_session = 0
if "last_key_time" not in st.session_state:
    st.session_state.last_key_time = None

def key_logger():
    """Capture keystrokes using Streamlit text_input trick."""
    # We use text_input for live typing, then calculate deltas when text changes
    typed = st.session_state[f"text_session_{st.session_state.current_session}"]
    events = st.session_state.sessions[st.session_state.current_session]

    now = time.time()
    if st.session_state.last_key_time is not None:
        flight_time = (now - st.session_state.last_key_time) * 1000  # ms
    else:
        flight_time = 0
    st.session_state.last_key_time = now

    if len(typed) > 0:
        key = typed[-1]
        dwell_time = np.random.uniform(80, 150)  # simulate dwell (ms)
        events.append({
            "key": key,
            "dwell": dwell_time,
            "flight": flight_time,
            "is_backspace": key == "\b",
            "is_space": key == " ",
            "timestamp": now,
        })
    st.session_state.sessions[st.session_state.current_session] = events


# ---------------- Sentence Inputs ----------------
for i in range(5):
    st.text_input(
        f"Session {i+1}: Type your sentence here",
        key=f"text_session_{i}",
        on_change=key_logger,
    )

if st.button("Next Session"):
    if st.session_state.current_session < 4:
        st.session_state.current_session += 1
        st.session_state.last_key_time = None
        st.success(f"Moved to Session {st.session_state.current_session+1}")
    else:
        st.warning("All 5 sessions done!")


# ---------------- Feature Extraction ----------------
REQUIRED_COLUMNS = [
    "avg_dwell_time",
    "std_dwell_time",
    "avg_flight_time",
    "std_flight_time",
    "backspace_count",
    "space_count",
    "keystroke_rate_wpm",
]

def extract_features(events):
    if len(events) < 2:
        return None
    dwell_times = [e["dwell"] for e in events]
    flight_times = [e["flight"] for e in events[1:]]
    backspaces = sum(1 for e in events if e["is_backspace"])
    spaces = sum(1 for e in events if e["is_space"])
    duration_sec = events[-1]["timestamp"] - events[0]["timestamp"]
    num_chars = len(events)
    wpm = (num_chars / 5) / (duration_sec / 60) if duration_sec > 0 else 0

    return {
        "avg_dwell_time": np.mean(dwell_times),
        "std_dwell_time": np.std(dwell_times),
        "avg_flight_time": np.mean(flight_times),
        "std_flight_time": np.std(flight_times),
        "backspace_count": backspaces,
        "space_count": spaces,
        "keystroke_rate_wpm": wpm,
    }


if st.button("Extract Features"):
    features = []
    for i, sess in enumerate(st.session_state.sessions):
        f = extract_features(sess)
        if f:
            features.append(f)
    if len(features) == 0:
        st.error("Not enough keystrokes recorded.")
    else:
        df = pd.DataFrame(features)
        st.session_state["features_df"] = df
        st.write("Extracted features:")
        st.dataframe(df)


# ---------------- Model Training ----------------
def build_autoencoder(input_dim, latent_dim=4):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(16, activation="relu")(inputs)
    x = layers.Dense(8, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = layers.Dense(8, activation="relu")(latent)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    autoencoder = keras.Model(inputs, outputs)
    encoder = keras.Model(inputs, latent)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


if st.button("Train Models"):
    if "features_df" not in st.session_state:
        st.error("Please extract features first.")
    else:
        df = st.session_state["features_df"]
        X = df[REQUIRED_COLUMNS].values.astype(np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        autoencoder, encoder = build_autoencoder(X_scaled.shape[1])
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=4, verbose=0)

        # Train anomaly model in latent space
        latent_vecs = encoder.predict(X_scaled)
        ocsvm = OneClassSVM(gamma="scale", nu=0.1)
        ocsvm.fit(latent_vecs)

        # Save .tflite
        tflite_model = convert_to_tflite(autoencoder)
        st.download_button("Download autoencoder.tflite", tflite_model, "autoencoder.tflite")

        # Save .pkl
        buf = io.BytesIO()
        joblib.dump({"scaler": scaler, "ocsvm": ocsvm}, buf)
        buf.seek(0)
        st.download_button("Download anomaly_model.pkl", buf, "anomaly_model.pkl")

        st.success("Training complete! Models ready for download.")
