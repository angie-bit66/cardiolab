"""
Módulo de utilidades para CardioLab.

Propósito: Proporcionar funciones auxiliares para visualización, 
gráficos y operaciones comunes en el análisis de señales ECG.

Relacionado con: Visualización de resultados y apoyo a otros módulos.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_ecg_matplotlib(signal, fs, title="ECG Signal"):
    """
    Grafica una señal ECG usando Matplotlib.
    
    Args:
        signal (array): Amplitudes de la señal ECG.
        fs (float): Frecuencia de muestreo en Hz.
        title (str): Título del gráfico.
    
    Returns:
        fig: Figura de Matplotlib.
    """
    time = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, signal, linewidth=0.8, color='navy')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude (mV)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_ecg_plotly(signal, fs, title="ECG Signal", show_peaks=None):
    """
    Grafica una señal ECG usando Plotly (interactivo).
    
    Args:
        signal (array): Amplitudes de la señal ECG.
        fs (float): Frecuencia de muestreo en Hz.
        title (str): Título del gráfico.
        show_peaks (array, optional): Índices de picos R para marcar.
    
    Returns:
        fig: Figura de Plotly.
    """
    time = np.arange(len(signal)) / fs
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time, y=signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='navy', width=2)
    ))
    
    if show_peaks is not None and len(show_peaks) > 0:
        peak_times = show_peaks / fs
        peak_values = signal[show_peaks]
        fig.add_trace(go.Scatter(
            x=peak_times, y=peak_values,
            mode='markers',
            name='R Peaks',
            marker=dict(color='red', size=8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (mV)',
        hovermode='x unified',
        height=450,
        template='plotly_white'
    )
    return fig


def plot_comparison_plotly(signal_before, signal_after, fs, title="ECG: Before vs After"):
    """
    Grafica comparación de señal antes y después del procesamiento.
    
    Args:
        signal_before (array): Señal original.
        signal_after (array): Señal procesada.
        fs (float): Frecuencia de muestreo.
        title (str): Título del gráfico.
    
    Returns:
        fig: Figura de Plotly con subplots.
    """
    time = np.arange(len(signal_before)) / fs
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Raw ECG Signal", "Filtered ECG Signal"),
        shared_xaxes=True
    )
    
    fig.add_trace(
        go.Scatter(x=time, y=signal_before, mode='lines', name='Raw', 
                   line=dict(color='lightcoral', width=1.5)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=time, y=signal_after, mode='lines', name='Filtered', 
                   line=dict(color='navy', width=1.5)),
        row=2, col=1
    )
    
    fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (mV)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    
    fig.update_layout(height=600, title_text=title, hovermode='x unified', template='plotly_white')
    return fig


def plot_rr_intervals(rr_intervals, title="R-R Intervals Over Time"):
    """
    Grafica los intervalos R-R en el tiempo.
    
    Args:
        rr_intervals (array): Intervalos R-R en milisegundos.
        title (str): Título del gráfico.
    
    Returns:
        fig: Figura de Plotly.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=rr_intervals,
        mode='lines+markers',
        name='R-R Intervals',
        line=dict(color='steelblue', width=2),
        marker=dict(size=5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Beat Number',
        yaxis_title='R-R Interval (ms)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    return fig


def plot_classification_confidence(predictions, class_names, confidence_scores):
    """
    Grafica la confianza de clasificación como gráfico de barras.
    
    Args:
        predictions (str): Clase predicha.
        class_names (list): Nombres de las clases.
        confidence_scores (list): Confianza para cada clase (0-1).
    
    Returns:
        fig: Figura de Plotly.
    """
    colors = ['red' if name == predictions else 'lightblue' for name in class_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=confidence_scores,
            marker_color=colors,
            text=[f'{score:.2%}' for score in confidence_scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Classification Confidence Scores',
        xaxis_title='Class',
        yaxis_title='Confidence',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    return fig


def create_metrics_table(metrics_dict):
    """
    Crea una tabla con métricas de diagnóstico.
    
    Args:
        metrics_dict (dict): Diccionario con métricas.
    
    Returns:
        str: Tabla formateada en HTML o texto.
    """
    rows = []
    for key, value in metrics_dict.items():
        rows.append(f"{key}: {value}")
    return "\n".join(rows)
