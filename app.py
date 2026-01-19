import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="NeuroVoice | Frailty Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Cool" Dark/Medical Look
st.markdown("""
    <style>
   .main {
        background-color: #0e1117;
    }
   .stApp {
        background: linear-gradient(to bottom right, #0e1117, #161b24);
    }
    h1 {
        color: #00e676; /* Medical Green */
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #ffffff;
    }
   .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00e676;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FEATURE EXTRACTOR (THE ENGINE)
# ==========================================
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

            # --- A1: Zero-Crossing Rate ---
            zcr = librosa.feature.zero_crossing_rate(y_trimmed)
            features['A1_zcr_mean'] = np.mean(zcr)
            features['A1_zcr_std'] = np.std(zcr)

            # --- A2: Shimmer ---
            try:
                pitch = sound.to_pitch()
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
                shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['A2_shimmer'] = shimmer
            except: features['A2_shimmer'] = 0

            # --- A3: Formants ---
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

            # --- A4: Energy Ratio ---
            S = np.abs(librosa.stft(y_trimmed))**2
            freqs = librosa.fft_frequencies(sr=sr)
            cutoff = np.argmin(np.abs(freqs - 800))
            features['A4_energy_ratio'] = np.sum(S[:cutoff, :]) / np.sum(S) if np.sum(S) > 0 else 0

            # --- MFCCs (Timbre) ---
            mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])

            return features
        except Exception as e:
            return None

# ==========================================
# 3. HELPER FUNCTIONS FOR UI
# ==========================================
def create_radar_chart(features):
    # Normalize values roughly for visualization (based on typical benchmarks)
    categories =
    
    # Fake normalization for visual scaling (e.g., Shimmer*100 to make it visible)
    values = [
        features.get('A2_shimmer', 0) * 1000, 
        features.get('A1_zcr_mean', 0) * 500,
        features.get('A3_f1_std', 0) / 5,
        features.get('A4_energy_ratio', 0) * 100
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Voice',
        line_color='#00e676'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=)),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': , 'y': },
        title = {'text': "Frailty Probability", 'font': {'color': 'white'}},
        number = {'suffix': "%", 'font': {'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#ff2b2b" if probability > 0.5 else "#00e676"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': , 'color': 'rgba(0, 230, 118, 0.3)'},
                {'range': , 'color': 'rgba(255, 43, 43, 0.3)'}],
        }))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    return fig

# ==========================================
# 4. MAIN APP LAYOUT
# ==========================================
st.title("🧠 NeuroVoice AI")
st.markdown("### Auditory Frailty Assessment System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.header("Instructions")
    st.info("1. Click 'Start Recording'.\n2. Say: 'The quick brown fox jumps over the lazy dog.'\n3. Click 'Analyze'.")
    st.warning("Ensure you are in a quiet room.")

# Main Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### 🎙️ Input Source")
    # Streamlit Native Audio Recorder
    audio_value = st.audio_input("Record Voice Sample")

    # Load Model
    if os.path.exists("frailty_model.pkl"):
        model = joblib.load("frailty_model.pkl")
    else:
        st.error("⚠️ Model file missing! Please upload 'frailty_model.pkl'.")
        model = None

with col2:
    if audio_value and model:
        # Save temp file
        with open("temp_input.wav", "wb") as f:
            f.write(audio_value.getbuffer())
        
        with st.spinner("🤖 Extracting Vocal Biomarkers..."):
            extractor = FrailtyFeatureExtractor()
            features = extractor.get_features("temp_input.wav")
            
            if features:
                # Predict
                df_input = pd.DataFrame([features])
                prediction = model.predict(df_input)
                proba = model.predict_proba(df_input)[1] # Probability of Frailty (Class 1)
                
                # --- RESULTS DASHBOARD ---
                st.markdown("### 🩺 Diagnostic Results")
                
                # Top Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Vocal Stability (Shimmer)", f"{features['A2_shimmer']:.4f}", delta_color="inverse")
                m2.metric("Articulation (F1)", f"{features['A3_f1_std']:.1f}")
                m3.metric("Energy Ratio", f"{features['A4_energy_ratio']:.2f}")
                
                # Visualizations
                viz1, viz2 = st.columns([1, 1])
                
                with viz1:
                    st.plotly_chart(create_gauge(proba), use_container_width=True)
                    if prediction == 1:
                        st.error("🔴 **Classification: SIGNS OF FRAILTY DETECTED**")
                    else:
                        st.success("🟢 **Classification: ROBUST / HEALTHY**")
                
                with viz2:
                    st.plotly_chart(create_radar_chart(features), use_container_width=True)
                
                # AI Explanation
                with st.expander("🔎 AI Analysis Explanation"):
                    st.write(f"""
                    The model analyzed **17 acoustic features**. 
                    - **Shimmer ({features['A2_shimmer']:.3f})**: High values indicate vocal instability (tremor/weakness).
                    - **MFCCs**: Analyzed the timbre and texture of the voice for slurring.
                    - **Probability Score**: The system is **{proba*100:.1f}%** confident in this result.
                    """)
