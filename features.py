"""
Módulo de extracción de características para CardioLab.

Propósito: Extraer características relevantes de las señales ECG,
incluyendo intervalos R-R, variabilidad de frecuencia cardíaca (HRV)
y otras métricas para diagnóstico de arritmias y apnea del sueño.

Relacionado con: Análisis de variabilidad cardíaca que distingue
entre ritmo normal, arritmias y episodios de apnea.
"""

import numpy as np
from scipy import signal, stats


def detect_r_peaks(ecg_signal, fs, threshold_factor=0.6):
    """
    Detecta los picos R en la señal ECG usando análisis de amplitud.
    
    Args:
        ecg_signal (array): Señal ECG preprocesada.
        fs (float): Frecuencia de muestreo en Hz.
        threshold_factor (float): Factor multiplicador del umbral.
    
    Returns:
        array: Índices de los picos R.
    """
    # Usar derivada para detectar transiciones rápidas
    derivative = np.gradient(ecg_signal)
    squared_derivative = derivative ** 2
    
    # Suavizar
    window = int(fs * 0.04)
    if window % 2 == 0:
        window += 1
    smoothed = signal.medfilt(squared_derivative, kernel_size=window)
    
    # Umbral adaptativo
    threshold = threshold_factor * np.max(smoothed)
    
    # Encontrar picos con distancia mínima entre ellos (evita duplicados)
    min_distance = int(fs * 0.4)  # Mínimo 400 ms entre latidos (150 bpm máx)
    # Detectar picos R positivos
    r_peaks_pos, _ = signal.find_peaks(smoothed, height=threshold, distance=min_distance)

# Detectar picos R negativos (por si la señal está invertida)
    r_peaks_neg, _ = signal.find_peaks(-smoothed, height=threshold, distance=min_distance)

# Escoger el tipo de pico más representativo (el que tenga más detecciones)
    r_peaks = r_peaks_pos if len(r_peaks_pos) >= len(r_peaks_neg) else r_peaks_neg

    return r_peaks


def calculate_rr_intervals(r_peaks, fs):
    """
    Calcula los intervalos R-R (tiempo entre picos R consecutivos).
    
    Args:
        r_peaks (array): Índices de los picos R.
        fs (float): Frecuencia de muestreo en Hz.
    
    Returns:
        array: Intervalos R-R en milisegundos.
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    # Diferencias entre picos consecutivos en muestras
    peak_differences = np.diff(r_peaks)
    
    # Convertir a milisegundos
    rr_intervals = (peak_differences / fs) * 1000
    
    return rr_intervals


def calculate_heart_rate(rr_intervals):
    """
    Calcula la frecuencia cardíaca promedio a partir de intervalos R-R.
    
    Args:
        rr_intervals (array): Intervalos R-R en milisegundos.
    
    Returns:
        float: Frecuencia cardíaca en latidos por minuto (bpm).
    """
    if len(rr_intervals) == 0:
        return 0.0
    
    mean_rr = np.mean(rr_intervals)
    heart_rate = 60000 / mean_rr  # Convertir ms a bpm
    
    return heart_rate


def calculate_hrv_time_domain(rr_intervals):
    """
    Calcula métricas de variabilidad cardíaca en el dominio del tiempo.
    
    Args:
        rr_intervals (array): Intervalos R-R en milisegundos.
    
    Returns:
        dict: Diccionario con métricas HRV:
            - SDNN: Desviación estándar de intervalos R-R
            - RMSSD: Raíz cuadrada de la media de diferencias al cuadrado
            - NN50: Número de diferencias > 50 ms
            - pNN50: Porcentaje de NN50
    """
    if len(rr_intervals) < 2:
        return {'SDNN': 0, 'RMSSD': 0, 'NN50': 0, 'pNN50': 0}
    
    # SDNN: Desviación estándar
    sdnn = np.std(rr_intervals, ddof=1)
    
    # RMSSD: Raíz cuadrada de media de diferencias al cuadrado
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # NN50 y pNN50: Diferencias > 50 ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    
    return {
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'NN50': nn50,
        'pNN50': pnn50
    }


def calculate_hrv_frequency_domain(rr_intervals, fs=4):
    """
    Calcula métricas de variabilidad cardíaca en el dominio de frecuencia.
    
    Args:
        rr_intervals (array): Intervalos R-R en milisegundos.
        fs (float): Frecuencia de muestreo del HRV en Hz (interpolación).
    
    Returns:
        dict: Diccionario con:
            - VLF: Muy baja frecuencia (0-0.04 Hz)
            - LF: Baja frecuencia (0.04-0.15 Hz)
            - HF: Alta frecuencia (0.15-0.4 Hz)
            - LF/HF: Ratio
    """
    if len(rr_intervals) < 10:
        return {'VLF': 0, 'LF': 0, 'HF': 0, 'LF/HF': 0}
    
    # Normalizar los intervalos R-R
    rr_normalized = rr_intervals - np.mean(rr_intervals)
    
    # Calcular FFT
    fft = np.fft.fft(rr_normalized)
    frequencies = np.fft.fftfreq(len(rr_normalized), 1/fs)
    power = np.abs(fft) ** 2
    
    # Filtrar solo frecuencias positivas
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    power = power[positive_freq_idx]
    
    # Definir bandas de frecuencia
    vlf_band = (frequencies > 0.0) & (frequencies <= 0.04)
    lf_band = (frequencies > 0.04) & (frequencies <= 0.15)
    hf_band = (frequencies > 0.15) & (frequencies <= 0.4)
    
    vlf = np.sum(power[vlf_band])
    lf = np.sum(power[lf_band])
    hf = np.sum(power[hf_band])
    
    lf_hf_ratio = lf / hf if hf > 0 else 0
    
    return {
        'VLF': vlf,
        'LF': lf,
        'HF': hf,
        'LF/HF': lf_hf_ratio
    }


def detect_arrhythmia_features(rr_intervals):
    """
    Detecta características indicativas de arritmias.
    
    Args:
        rr_intervals (array): Intervalos R-R en milisegundos.
    
    Returns:
        dict: Características de arritmia:
            - irregular_beats: Número de variaciones > 20%
            - mean_rr_deviation: Desviación promedio de RR
            - rr_std_dev: Desviación estándar
    """
    if len(rr_intervals) < 2:
        return {'irregular_beats': 0, 'mean_rr_deviation': 0, 'rr_std_dev': 0}
    
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    
    # Contar intervalos irregulares (fuera de ±20% de la media)
    lower_bound = mean_rr * 0.8
    upper_bound = mean_rr * 1.2
    irregular = np.sum((rr_intervals < lower_bound) | (rr_intervals > upper_bound))
    
    # Desviación promedio
    mean_deviation = np.mean(np.abs(rr_intervals - mean_rr))
    
    return {
        'irregular_beats': int(irregular),
        'mean_rr_deviation': mean_deviation,
        'rr_std_dev': std_rr
    }


def detect_apnea_features(rr_intervals):
    """
    Detecta características indicativas de apnea del sueño.
    
    Args:
        rr_intervals (array): Intervalos R-R en milisegundos.
    
    Returns:
        dict: Características de apnea:
            - apnea_index: Índice calculado (0-100)
            - bradycardia_events: Eventos de bradicardia
            - heart_rate_variability: Variabilidad de frecuencia
    """
    if len(rr_intervals) < 5:
        return {'apnea_index': 0, 'bradycardia_events': 0, 'heart_rate_variability': 0}
    
    # Calcular cambios en RR
    rr_diff = np.abs(np.diff(rr_intervals))
    mean_rr = np.mean(rr_intervals)
    
    # Eventos de bradicardia (RR > 1000 ms = HR < 60 bpm)
    bradycardia = np.sum(rr_intervals > 1000)
    
    # Variabilidad de cambios rápidos en RR (característica de apnea)
    rapid_changes = np.sum(rr_diff > 200)
    
    # Índice de apnea simulado
    apnea_index = (bradycardia + rapid_changes) / len(rr_intervals) * 100
    apnea_index = np.clip(apnea_index, 0, 100)
    
    hrv = np.std(rr_intervals)
    
    return {
        'apnea_index': apnea_index,
        'bradycardia_events': int(bradycardia),
        'heart_rate_variability': hrv
    }


def extract_all_features(ecg_signal, r_peaks, fs):
    """
    Extrae todas las características de una señal ECG procesada.
    
    Args:
        ecg_signal (array): Señal ECG preprocesada.
        r_peaks (array): Índices de picos R.
        fs (float): Frecuencia de muestreo en Hz.
    
    Returns:
        dict: Diccionario con todas las características extraídas.
    """
    rr_intervals = calculate_rr_intervals(r_peaks, fs)
    heart_rate = calculate_heart_rate(rr_intervals)
    hrv_time = calculate_hrv_time_domain(rr_intervals)
    hrv_freq = calculate_hrv_frequency_domain(rr_intervals)
    arrhythmia_feat = detect_arrhythmia_features(rr_intervals)
    apnea_feat = detect_apnea_features(rr_intervals)
    
    return {
        'heart_rate': heart_rate,
        'rr_intervals': rr_intervals,
        'hrv_time_domain': hrv_time,
        'hrv_frequency_domain': hrv_freq,
        'arrhythmia_features': arrhythmia_feat,
        'apnea_features': apnea_feat,
        'num_r_peaks': len(r_peaks)
    }

def create_feature_vector(features_dict):
    """
    Convierte el diccionario de características en un vector numérico
    que puede usarse para el modelo de clasificación (arritmia/apnea).

    Args:
        features_dict (dict): Diccionario de características generado por extract_all_features().

    Returns:
        np.array: Vector de características.
    """
    return np.array([
        features_dict['heart_rate'],
        features_dict['hrv_time_domain']['SDNN'],
        features_dict['hrv_time_domain']['RMSSD'],
        features_dict['hrv_frequency_domain']['LF'],
        features_dict['hrv_frequency_domain']['HF'],
        features_dict['arrhythmia_features']['rr_std_dev'],
        features_dict['apnea_features']['apnea_index']
    ]).reshape(1, -1)
