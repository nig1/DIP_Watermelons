import librosa
import numpy as np
import pandas as pd

# Function to preprocess audio
def preprocess_audio(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features: Max amplitude frequency, spectral centroid, spectral bandwidth
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    max_ampl_freq = np.argmax(np.abs(np.fft.fft(y)))

    # Example feature vector (max ampl freq, spectral centroid, spectral bandwidth)
    audio_features = [max_ampl_freq, spectral_centroid, spectral_bandwidth]

    return np.array(audio_features)
