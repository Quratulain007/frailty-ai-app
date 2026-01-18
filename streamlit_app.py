import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call

# ==========================================
# 1. SETUP & HELPER CLASSES
# ==========================================
# We must include the EXACT same class used during training
class FrailtyFeatureExtractor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def get_features(self, file_path):
        try:
            # Load with Librosa
            y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) < self.target_sr * 0.5: return None
            
            # Load with Parselmouth
            sound = parselmouth.Sound(file_path)
            features = {}

            # A1: ZCR
            zcr = librosa.feature.zero_crossing_rate(y_trimmed)
            features['A1_zcr_mean'] = np.mean(zcr)
            features['A1_zcr_std'] = np.std(zcr)

            # A2: Shimmer
            try:
                pitch = sound.to_pitch()
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
                shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['A2_shimmer'] = shimmer
            except: features['A2_shimmer'] = 0

            # A3: Formants
            try:
                formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
                f1_list =
                num_frames = call(formant, "Get number of frames")
                for t in range(1, num_frames + 1):
                    time = call(formant, "Get time from frame number", t)
                    f1 = call(formant, "Get value at time", 1, time, 'Hertz', 'Linear')
                    if not np.isnan(f1): f1_list.append(f1)
                features['A3_f1_std'] = np.std(f1_list) if f1_list else 0
            except: features['A3_f1_std'] = 0

            # A4: Energy Ratio
            S = np.abs(librosa.stft(y_trimmed))**2
            freqs = librosa.fft_frequencies(sr=sr)
            cutoff = np.argmin(np.abs(freqs - 800))
            features['A4_energy_ratio'] = np.sum(S[:cutoff, :]) / np.sum(S) if np.sum(S) > 0 else 0

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])

            return features
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return None

# ==========================================
# 2. STREAMLIT APP INTERFACE
# ==========================================
st.set_page_config(page_title="Frailty AI Assessment", page_icon="🩺")

st.title("🩺 AI Auditory Frailty Assessment")
st.write("Record a short sentence (e.g., 'The quick brown fox jumps over the lazy dog') to assess vocal biomarkers associated with frailty.")

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists("frailty_model.pkl"):
        return joblib.load("frailty_model.pkl")
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model file 'frailty_model.pkl' not found. Please upload it to your GitHub repository.")
else:
    # New Native Audio Input (Streamlit 1.37+)
    audio_value = st.audio_input("Record your voice")

    if audio_value:
        st.audio(audio_value)
        
        # Save to temp file for processing
        with open("temp_input.wav", "wb") as f:
            f.write(audio_value.getbuffer())
        
        with st.spinner("Analyzing vocal biomarkers..."):
            extractor = FrailtyFeatureExtractor()
            features = extractor.get_features("temp_input.wav")
            
            if features:
                # Prepare data for model
                df_input = pd.DataFrame([features])
                
                # Predict
                prediction = model.predict(df_input)
                probability = model.predict_proba(df_input)
                
                # Display Results
                st.divider()
                if prediction == 1:
                    st.error(f"🔴 Result: SIGNS OF FRAILTY DETECTED (Confidence: {probability[1]*100:.1f}%)")
                    st.write("The model detected acoustic patterns (shimmer, variability) consistent with the frailty phenotype.")
                else:
                    st.success(f"🟢 Result: ROBUST / HEALTHY (Confidence: {probability*100:.1f}%)")
                    st.write("Your vocal markers are within the healthy range for this model.")
                
                # Show Data Table
                with st.expander("See Detailed Biomarkers"):
                    st.json(features)
