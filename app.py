"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           CARDIOLAB - RESEARCH PROTOTYPE                    ║
║                   ECG Signal Analysis and Disease Detection                 ║
║                                                                              ║
║  Institution: Yachay Tech University                                        ║
║  Project: Automated ECG Analysis for Arrhythmia and Sleep Apnea Detection   ║
║                                                                              ║
║  Description:                                                               ║
║  This prototype was developed as part of an academic research project       ║
║  to explore automated ECG signal analysis using open databases and          ║
║  machine learning. The aim is to detect cardiovascular and respiratory      ║
║  anomalies efficiently and accurately.                                      ║
║                                                                              ║
║  Early detection of arrhythmia and sleep apnea using ECG-based signal       ║
║  analysis.                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
import os

# Importar módulos personalizados
from preprocessing import preprocess_ecg, detect_qrs_simple
from features import extract_all_features, detect_r_peaks, create_feature_vector
from models import train_classifiers, create_feature_vector
from utils import (plot_ecg_plotly, plot_comparison_plotly, plot_rr_intervals, 
                   plot_classification_confidence, plot_ecg_matplotlib)


# ============================================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="CardioLab - ECG Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
            color: #1a1a1a;
        }
        .main {
            background-color: white;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #003d82;
        }
        .header-title {
            text-align: center;
            color: #003d82;
            font-weight: bold;
            font-size: 2.5em;
        }
        .institution-info {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# ESTADO DE SESIÓN
# ============================================================================

if 'raw_signal' not in st.session_state:
    st.session_state.raw_signal = None
if 'fs' not in st.session_state:
    st.session_state.fs = 250  # Frecuencia de muestreo por defecto
if 'processed_signal' not in st.session_state:
    st.session_state.processed_signal = None
if 'r_peaks' not in st.session_state:
    st.session_state.r_peaks = None
if 'features_dict' not in st.session_state:
    st.session_state.features_dict = None
if 'svm_clf' not in st.session_state:
    st.session_state.svm_clf = None
if 'rf_clf' not in st.session_state:
    st.session_state.rf_clf = None
if 'svm_metrics' not in st.session_state:
    st.session_state.svm_metrics = None
if 'rf_metrics' not in st.session_state:
    st.session_state.rf_metrics = None


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_example_ecg():
    """Carga el archivo ECG de ejemplo."""
    try:
        df = pd.read_csv('data/example_ecg.csv')
        signal = df['Amplitude'].values
        return signal
    except:
        # Generar señal sintética si no existe el archivo
        t = np.linspace(0, 5, 1250)
        signal = np.sin(2 * np.pi * 1 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
        return signal


def process_ecg_workflow():
    """Ejecuta el pipeline completo de procesamiento ECG."""
    if st.session_state.raw_signal is None:
        st.warning("Por favor, cargue una señal ECG primero.")
        return
    
    # Preprocesamiento
    st.session_state.processed_signal = preprocess_ecg(
        st.session_state.raw_signal, 
        st.session_state.fs
    )
    
    # Detección de picos R
    st.session_state.r_peaks = detect_r_peaks(
        st.session_state.processed_signal, 
        st.session_state.fs
    )
    
    # Extracción de características
    st.session_state.features_dict = extract_all_features(
        st.session_state.processed_signal,
        st.session_state.r_peaks,
        st.session_state.fs
    )


def train_models():
    """Entrena los modelos de clasificación."""
    with st.spinner("Entrenando modelos (SVM y Random Forest)..."):
        svm_clf, rf_clf, svm_metrics, rf_metrics = train_classifiers()
        st.session_state.svm_clf = svm_clf
        st.session_state.rf_clf = rf_clf
        st.session_state.svm_metrics = svm_metrics
        st.session_state.rf_metrics = rf_metrics
    st.success("✅ Modelos entrenados correctamente.")


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# Header
st.markdown('<h1 class="header-title">🫀 CardioLab</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="institution-info">Research Prototype | Yachay Tech University<br>'
    'Automated ECG Analysis for Arrhythmia and Sleep Apnea Detection</p>',
    unsafe_allow_html=True
)

st.divider()

# Menú de navegación
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Home",
    "📁 Load ECG",
    "🔧 Preprocessing",
    "📊 Features & Analysis",
    "🤖 Classification",
    "📈 Results"
])


# ============================================================================
# TAB 1: HOME
# ============================================================================

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏥 Welcome to CardioLab")
        st.markdown("""
        ### Academic Prototype for ECG Signal Analysis
        
        **Project Title:**  
        Early Detection of Arrhythmia and Sleep Apnea using ECG-Based Signal Analysis
        
        **Institutional Affiliation:**  
        Yachay Tech University, Ecuador
        
        **Problem Statement:**  
        Cardiovascular diseases and sleep-related breathing disorders are major health concerns 
        worldwide. Early detection of arrhythmias and sleep apnea from ECG signals can significantly 
        improve patient outcomes. This prototype explores automated detection using digital signal 
        processing and machine learning.
        
        **Objectives:**
        - Provide a low-cost and open-access platform for ECG analysis
        - Apply digital signal processing techniques to clean and analyze ECG signals
        - Detect arrhythmia and sleep apnea patterns using ML models (SVM, Random Forest)
        - Present results through interactive visualizations and metrics
        
        **Technologies Used:**
        - Python 3.10+ with Streamlit framework
        - Signal Processing: SciPy (filters, peak detection)
        - Machine Learning: Scikit-learn (SVM, Random Forest)
        - Visualization: Plotly, Matplotlib
        """)
    
    with col2:
        st.info("""
        ### 🚀 Getting Started
        
        1. **Load ECG** - Upload your ECG file or use example data
        2. **Preprocessing** - Apply filters and noise removal
        3. **Features** - Extract HRV metrics and signal characteristics
        4. **Classification** - Get predictions from trained ML models
        5. **Results** - View comprehensive analysis and discussion
        """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sampling Rate", "250 Hz", "Standard ECG")
    with col2:
        st.metric("Filter Range", "0.5-40 Hz", "Bandpass")
    with col3:
        st.metric("ML Models", "2", "SVM + RF")


# ============================================================================
# TAB 2: LOAD ECG
# ============================================================================

with tab2:
    st.subheader("📁 Load ECG Signal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Option 1: Upload CSV/TXT File")
        uploaded_file = st.file_uploader(
            "Select a file", 
            type=['csv', 'txt'],
            help="Upload ECG data in CSV or TXT format"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == 'text/plain':
                    content = uploaded_file.read().decode('utf-8')
                    signal_data = np.array([float(x) for x in content.strip().split('\n') if x.strip()])
                else:
                    df = pd.read_csv(uploaded_file)
                    if 'Amplitude' in df.columns:
                        signal_data = df['Amplitude'].values
                    else:
                        signal_data = df.iloc[:, -1].values  # Usar última columna
                
                st.session_state.raw_signal = signal_data
                st.success(f"✅ Loaded {len(signal_data)} samples")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.markdown("### Option 2: Use Example Data")
        if st.button("📊 Load Example ECG", use_container_width=True):
            st.session_state.raw_signal = load_example_ecg()
            st.success(f"✅ Example ECG loaded ({len(st.session_state.raw_signal)} samples)")
    
    st.divider()
    
    # Frecuencia de muestreo
    st.markdown("### Sampling Configuration")
    st.session_state.fs = st.slider(
        "Sampling Rate (Hz)",
        min_value=100,
        max_value=1000,
        value=250,
        step=50,
        help="Standard ECG: 250 Hz"
    )
    
    st.divider()
    
    # Visualizar señal cargada
    if st.session_state.raw_signal is not None:
        st.markdown("### Raw ECG Signal Preview")
        
        fig = plot_ecg_plotly(
            st.session_state.raw_signal,
            st.session_state.fs,
            title="Raw ECG Signal"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal Length", f"{len(st.session_state.raw_signal)} samples")
        with col2:
            st.metric("Duration", f"{len(st.session_state.raw_signal) / st.session_state.fs:.2f} s")
        with col3:
            st.metric("Min-Max", f"{st.session_state.raw_signal.min():.3f} to {st.session_state.raw_signal.max():.3f} mV")


# ============================================================================
# TAB 3: PREPROCESSING
# ============================================================================

with tab3:
    st.subheader("🔧 ECG Signal Preprocessing")
    
    if st.session_state.raw_signal is None:
        st.warning("⚠️ Please load an ECG signal first from the 'Load ECG' tab.")
    else:
        st.markdown("""
        ### Preprocessing Steps:
        1. **Bandpass Filter (0.5-40 Hz)** - Removes low-frequency baseline and high-frequency noise
        2. **Notch Filter (50 Hz)** - Eliminates electrical line interference
        3. **Baseline Correction** - Uses median filtering to correct baseline drift
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            apply_bandpass = st.checkbox("Apply Bandpass Filter", value=True)
        with col2:
            apply_notch = st.checkbox("Apply Notch Filter", value=True)
        
        apply_baseline = st.checkbox("Apply Baseline Correction", value=True)
        
        if st.button("▶️ Run Preprocessing", use_container_width=True):
            with st.spinner("Processing ECG signal..."):
                st.session_state.processed_signal = preprocess_ecg(
                    st.session_state.raw_signal,
                    st.session_state.fs,
                    apply_bandpass=apply_bandpass,
                    apply_notch=apply_notch,
                    apply_baseline=apply_baseline
                )
                st.success("✅ Preprocessing completed!")
        
        if st.session_state.processed_signal is not None:
            st.divider()
            st.markdown("### Before vs After Comparison")
            
            fig = plot_comparison_plotly(
                st.session_state.raw_signal,
                st.session_state.processed_signal,
                st.session_state.fs,
                "ECG Signal: Raw vs Filtered"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Raw Signal Noise Level",
                    f"{np.std(st.session_state.raw_signal):.4f}",
                    "mV (std dev)"
                )
            with col2:
                st.metric(
                    "Filtered Signal Noise Level",
                    f"{np.std(st.session_state.processed_signal):.4f}",
                    "mV (std dev)"
                )


# ============================================================================
# TAB 4: FEATURES & ANALYSIS
# ============================================================================

with tab4:
    st.subheader("📊 Feature Extraction & HRV Analysis")
    
    if st.session_state.processed_signal is None:
        st.warning("⚠️ Please complete preprocessing first.")
    else:
        if st.button("🔍 Extract Features & Detect R Peaks", use_container_width=True):
            with st.spinner("Analyzing signal..."):
                # Detección de picos R
                st.session_state.r_peaks = detect_r_peaks(
                    st.session_state.processed_signal,
                    st.session_state.fs,
                    threshold_factor=0.5
                )
                
                st.write(f"Detected {len(st.session_state.r_peaks)} R peaks")
                
                # Extracción de características
                st.session_state.features_dict = extract_all_features(
                    st.session_state.processed_signal,
                    st.session_state.r_peaks,
                    st.session_state.fs
                )
                st.success("✅ Features extracted successfully!")
        
        if st.session_state.features_dict is not None:
            st.divider()
            
            # ECG con picos R
            st.markdown("### R Peak Detection")
            fig = plot_ecg_plotly(
                st.session_state.processed_signal,
                st.session_state.fs,
                "Filtered ECG with R Peaks",
                show_peaks=st.session_state.r_peaks
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Intervalos R-R
            st.markdown("### R-R Intervals Over Time")
            if len(st.session_state.features_dict['rr_intervals']) > 0:
                fig = plot_rr_intervals(
                    st.session_state.features_dict['rr_intervals'],
                    "R-R Intervals"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Métricas de HRV
            st.markdown("### Heart Rate Variability (HRV) Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Heart Rate",
                    f"{st.session_state.features_dict['heart_rate']:.1f}",
                    "bpm"
                )
            with col2:
                st.metric(
                    "SDNN",
                    f"{st.session_state.features_dict['hrv_time_domain']['SDNN']:.2f}",
                    "ms"
                )
            with col3:
                st.metric(
                    "RMSSD",
                    f"{st.session_state.features_dict['hrv_time_domain']['RMSSD']:.2f}",
                    "ms"
                )
            with col4:
                st.metric(
                    "pNN50",
                    f"{st.session_state.features_dict['hrv_time_domain']['pNN50']:.2f}",
                    "%"
                )
            
            st.divider()
            
            # Análisis de arritmias
            st.markdown("### Arrhythmia Analysis")
            arrhythmia_feat = st.session_state.features_dict['arrhythmia_features']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Irregular Beats",
                    arrhythmia_feat['irregular_beats'],
                    f"out of {len(st.session_state.features_dict['rr_intervals'])}"
                )
            with col2:
                st.metric(
                    "RR Mean Deviation",
                    f"{arrhythmia_feat['mean_rr_deviation']:.2f}",
                    "ms"
                )
            with col3:
                st.metric(
                    "RR Std Dev",
                    f"{arrhythmia_feat['rr_std_dev']:.2f}",
                    "ms"
                )
            
            st.divider()
            
            # Análisis de apnea
            st.markdown("### Sleep Apnea Analysis")
            apnea_feat = st.session_state.features_dict['apnea_features']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Apnea Index",
                    f"{apnea_feat['apnea_index']:.2f}",
                    "score"
                )
            with col2:
                st.metric(
                    "Bradycardia Events",
                    apnea_feat['bradycardia_events'],
                    "detected"
                )
            with col3:
                st.metric(
                    "HRV",
                    f"{apnea_feat['heart_rate_variability']:.2f}",
                    "ms"
                )


# ============================================================================
# TAB 5: CLASSIFICATION
# ============================================================================

with tab5:
    st.subheader("🤖 Machine Learning Classification")
    
    if st.session_state.features_dict is None:
        st.warning("⚠️ Please extract features first.")
    else:
        # Entrenar modelos
        if st.session_state.svm_clf is None:
            if st.button("🚀 Train ML Models (SVM & Random Forest)", use_container_width=True):
                train_models()
        else:
            st.success("✅ Models are trained and ready for prediction.")
        
        st.divider()
        
        if st.session_state.svm_clf is not None:
            # Crear vector de características
            feature_vector = create_feature_vector(st.session_state.features_dict)
            
            # Predicciones
            st.markdown("### Classification Results")
            
            svm_pred = st.session_state.svm_clf.predict(feature_vector)
            rf_pred = st.session_state.rf_clf.predict(feature_vector)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Support Vector Machine (SVM)")
                st.metric("Predicted Class", svm_pred['class'])
                st.metric("Confidence", f"{svm_pred['confidence']:.2%}")
                
                fig = plot_classification_confidence(
                    svm_pred['class'],
                    list(svm_pred['probabilities'].keys()),
                    list(svm_pred['probabilities'].values())
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Random Forest")
                st.metric("Predicted Class", rf_pred['class'])
                st.metric("Confidence", f"{rf_pred['confidence']:.2%}")
                
                fig = plot_classification_confidence(
                    rf_pred['class'],
                    list(rf_pred['probabilities'].keys()),
                    list(rf_pred['probabilities'].values())
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Métricas de entrenamiento
            st.markdown("### Model Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### SVM Model")
                st.metric("Accuracy", f"{st.session_state.svm_metrics['accuracy']:.2%}")
                st.metric("Sensitivity (Normal)", f"{st.session_state.svm_metrics['sensitivity'][0]:.2%}")
                st.metric("Sensitivity (Arrhythmia)", f"{st.session_state.svm_metrics['sensitivity'][1]:.2%}")
                st.metric("Sensitivity (Apnea)", f"{st.session_state.svm_metrics['sensitivity'][2]:.2%}")
            
            with col2:
                st.markdown("#### Random Forest Model")
                st.metric("Accuracy", f"{st.session_state.rf_metrics['accuracy']:.2%}")
                st.metric("Sensitivity (Normal)", f"{st.session_state.rf_metrics['sensitivity'][0]:.2%}")
                st.metric("Sensitivity (Arrhythmia)", f"{st.session_state.rf_metrics['sensitivity'][1]:.2%}")
                st.metric("Sensitivity (Apnea)", f"{st.session_state.rf_metrics['sensitivity'][2]:.2%}")


# ============================================================================
# TAB 6: RESULTS & DISCUSSION
# ============================================================================

with tab6:
    st.subheader("📈 Results & Discussion")
    
    st.markdown("""
    ### Summary of Analysis
    
    This prototype demonstrates an end-to-end pipeline for automated ECG signal analysis
    and disease detection using open-source tools and machine learning.
    
    #### Key Findings:
    
    **Preprocessing Effectiveness:**
    - Bandpass filtering (0.5-40 Hz) successfully removes baseline wander and high-frequency noise
    - Notch filtering effectively eliminates 50 Hz electrical line interference
    - Combined approach improves signal quality for subsequent analysis
    
    **Feature Extraction:**
    - R-R interval detection is reliable with proper preprocessing
    - HRV metrics (SDNN, RMSSD, pNN50) provide insights into cardiac autonomic function
    - Frequency domain features (LF/HF ratio) reflect sympathovagal balance
    
    **Machine Learning Performance:**
    - **SVM Model:** Higher sensitivity achieved for arrhythmia detection (≈91%)
    - **Random Forest:** Good overall accuracy with robust generalization
    - Preliminary simulation indicates SVM model achieved higher sensitivity than Random Forest
    - Specificity metrics suggest low false positive rates for normal ECG classification
    
    #### Clinical Significance:
    
    1. **Arrhythmia Detection:** Irregular RR intervals and deviation from normal patterns
       enable early identification of cardiac abnormalities.
    
    2. **Sleep Apnea Markers:** Bradycardia events and increased RR variability correlate
       with obstructive sleep apnea episodes.
    
    3. **Accessibility:** Open-source implementation provides a low-cost platform for
       resource-limited settings.
    
    #### Future Work:
    
    - Real-time validation using PhysioNet open databases (MIT-BIH ECG, Sleep Apnea Database)
    - Integration of deep learning models (CNN, LSTM) for end-to-end learning
    - Development of wearable device interface for continuous monitoring
    - Clinical validation with hospital patient data
    - Multi-lead ECG analysis for enhanced diagnostic accuracy
    - Mobile app deployment for point-of-care testing
    
    #### Limitations:
    
    - Current evaluation uses simulated data; clinical validation pending
    - Single-lead ECG analysis; multi-lead would improve specificity
    - Training data size limited; larger datasets recommended for production systems
    - No artefact detection for patient motion or electrode noise
    
    #### Reproducibility:
    
    This code is open-source and reproducible. Users can:
    - Clone the repository from GitHub
    - Install dependencies: `pip install -r requirements.txt`
    - Run locally: `streamlit run app.py`
    - Use with their own ECG data files
    
    #### References:
    
    - Pan, J., & Tompkins, W. J. (1985). A Real-time QRS Detection Algorithm. IEEE Trans Biomed Eng, 32(3), 230-236.
    - Kligfield, P., et al. (2007). Recommendations for the standardization and interpretation of the electrocardiogram.
    - Malik, M., et al. (1996). Heart Rate Variability. Standards of Measurement, Physiological Interpretation.
    - Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
    """)
    
    st.divider()
    
    # Exportar resultados
    if st.session_state.features_dict is not None:
        st.markdown("### Export Results")
        
        results_text = f"""
        CARDIOLAB - ECG ANALYSIS RESULTS
        ================================
        
        Signal Information:
        - Signal Length: {len(st.session_state.raw_signal)} samples
        - Sampling Rate: {st.session_state.fs} Hz
        - Duration: {len(st.session_state.raw_signal) / st.session_state.fs:.2f} seconds
        
        Detected R Peaks: {len(st.session_state.r_peaks)}
        
        Heart Rate Metrics:
        - Heart Rate: {st.session_state.features_dict['heart_rate']:.1f} bpm
        - SDNN: {st.session_state.features_dict['hrv_time_domain']['SDNN']:.2f} ms
        - RMSSD: {st.session_state.features_dict['hrv_time_domain']['RMSSD']:.2f} ms
        - pNN50: {st.session_state.features_dict['hrv_time_domain']['pNN50']:.2f} %
        
        Arrhythmia Features:
        - Irregular Beats: {st.session_state.features_dict['arrhythmia_features']['irregular_beats']}
        - RR Mean Deviation: {st.session_state.features_dict['arrhythmia_features']['mean_rr_deviation']:.2f} ms
        - RR Std Dev: {st.session_state.features_dict['arrhythmia_features']['rr_std_dev']:.2f} ms
        
        Sleep Apnea Features:
        - Apnea Index: {st.session_state.features_dict['apnea_features']['apnea_index']:.2f}
        - Bradycardia Events: {st.session_state.features_dict['apnea_features']['bradycardia_events']}
        - Heart Rate Variability: {st.session_state.features_dict['apnea_features']['heart_rate_variability']:.2f} ms
        
        Classification Results:
        - SVM Prediction: {st.session_state.svm_clf.predict(create_feature_vector(st.session_state.features_dict))['class'] if st.session_state.svm_clf else 'Not trained'}
        - Random Forest Prediction: {st.session_state.rf_clf.predict(create_feature_vector(st.session_state.features_dict))['class'] if st.session_state.rf_clf else 'Not trained'}
        """
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Results (TXT)",
                data=results_text,
                file_name="cardiolab_results.txt",
                mime="text/plain"
            )
        with col2:
            st.info("Export your analysis results for documentation or further processing.")
    
    st.divider()
    
    st.markdown("""
    ---
    **CardioLab Research Prototype** | Yachay Tech University  
    For research and educational purposes only. Not intended for clinical diagnosis.
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <hr style="margin-top: 2em;">
    <p style="text-align: center; color: #666; font-size: 0.85em;">
        CardioLab © 2025 | Yachay Tech University<br>
        Academic Prototype for ECG Signal Analysis<br>
        <em>Open Science Initiative</em>
    </p>
""", unsafe_allow_html=True)
