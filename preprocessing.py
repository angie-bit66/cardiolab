"""
Módulo de preprocesamiento para CardioLab.

Propósito: Aplicar filtros y técnicas de eliminación de ruido
a las señales ECG para mejorar la calidad antes del análisis.

Relacionado con: Limpieza de artefactos, ruido de línea eléctrica 
y desviación de línea base que afectan la detección de arritmias.
"""

import numpy as np
from scipy import signal


def bandpass_filter(signal_data, fs, lowcut=0.5, highcut=40, order=5):
    """
    Aplica un filtro paso banda (0.5-40 Hz) para eliminar ruido de baja frecuencia
    y artefactos de alta frecuencia.
    
    Args:
        signal_data (array): Señal ECG original.
        fs (float): Frecuencia de muestreo en Hz.
        lowcut (float): Frecuencia de corte baja en Hz.
        highcut (float): Frecuencia de corte alta en Hz.
        order (int): Orden del filtro.
    
    Returns:
        array: Señal filtrada.
    """
    # Diseñar el filtro Butterworth paso banda
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Evitar errores si las frecuencias están fuera del rango
    low = np.clip(low, 0.001, 0.999)
    high = np.clip(high, 0.001, 0.999)
    
    if low >= high:
        high = high * 1.1
        high = np.clip(high, 0.001, 0.999)
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered


def notch_filter(signal_data, fs, freq=50, quality=30):
    """
    Aplica un filtro notch para eliminar la interferencia de línea eléctrica (50/60 Hz).
    
    Args:
        signal_data (array): Señal ECG.
        fs (float): Frecuencia de muestreo en Hz.
        freq (float): Frecuencia a filtrar (50 Hz para Europa/América Latina, 60 para USA).
        quality (float): Factor de calidad del filtro.
    
    Returns:
        array: Señal sin interferencia de línea.
    """
    nyquist = fs / 2
    w0 = freq / nyquist
    
    # Evitar errores
    w0 = np.clip(w0, 0.001, 0.999)
    
    b, a = signal.iirnotch(w0, quality)
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered


def baseline_correction(signal_data, fs, window_length=None):
    """
    Corrige la desviación de línea base usando sustracción de media móvil.
    
    Args:
        signal_data (array): Señal ECG.
        fs (float): Frecuencia de muestreo en Hz.
        window_length (int, optional): Longitud de la ventana en muestras.
    
    Returns:
        array: Señal con línea base corregida.
    """
    if window_length is None:
        # Usar una ventana de ~500 ms
        window_length = int(fs * 0.5)
    
    # Asegurar que window_length es impar
    if window_length % 2 == 0:
        window_length += 1
    
    # Calcular la línea base usando media móvil
    baseline = signal.medfilt(signal_data, kernel_size=window_length)
    corrected = signal_data - baseline
    
    return corrected


def preprocess_ecg(signal_data, fs, apply_bandpass=True, apply_notch=True, 
                   apply_baseline=True, bandpass_params=None, notch_params=None):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    
    Args:
        signal_data (array): Señal ECG original.
        fs (float): Frecuencia de muestreo en Hz.
        apply_bandpass (bool): Aplicar filtro paso banda.
        apply_notch (bool): Aplicar filtro notch.
        apply_baseline (bool): Aplicar corrección de línea base.
        bandpass_params (dict): Parámetros para bandpass_filter.
        notch_params (dict): Parámetros para notch_filter.
    
    Returns:
        array: Señal preprocesada.
    """
    processed = signal_data.copy()
    
    if apply_bandpass:
        params = bandpass_params or {'lowcut': 0.5, 'highcut': 40, 'order': 5}
        processed = bandpass_filter(processed, fs, **params)
    
    if apply_notch:
        params = notch_params or {'freq': 50, 'quality': 30}
        processed = notch_filter(processed, fs, **params)
    
    if apply_baseline:
        processed = baseline_correction(processed, fs)
    
    return processed


def normalize_signal(signal_data):
    """
    Normaliza la señal al rango [0, 1].
    
    Args:
        signal_data (array): Señal ECG.
    
    Returns:
        array: Señal normalizada.
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    
    if max_val == min_val:
        return signal_data
    
    return (signal_data - min_val) / (max_val - min_val)


def detect_qrs_simple(signal_data, fs, threshold_factor=0.5):
    """
    Detección simple de complejos QRS usando diferenciación y umbralización.
    
    Args:
        signal_data (array): Señal ECG preprocesada.
        fs (float): Frecuencia de muestreo en Hz.
        threshold_factor (float): Factor de umbral.
    
    Returns:
        array: Índices de picos R detectados.
    """
    # Derivada de la señal para detectar cambios rápidos
    derivative = np.diff(signal_data)
    
    # Elevar al cuadrado para enfatizar picos
    squared = derivative ** 2
    
    # Aplicar suavizado
    window = int(fs * 0.05)  # Ventana de 50 ms
    if window % 2 == 0:
        window += 1
    smoothed = signal.medfilt(squared, kernel_size=window)
    
    # Calcular umbral adaptativo
    threshold = threshold_factor * np.mean(smoothed)
    
    # Encontrar picos por encima del umbral
    peaks, _ = signal.find_peaks(smoothed, height=threshold, distance=int(fs * 0.4))
    
    return peaks
