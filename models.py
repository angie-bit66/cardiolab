"""
Módulo de modelos de clasificación para CardioLab.

Propósito: Entrenar y aplicar modelos de aprendizaje automático
(SVM, Random Forest) para clasificar señales ECG como normales,
con arritmias o con apnea del sueño.

Relacionado con: Diagnóstico automático de condiciones cardiovasculares
y respiratorias basado en características extraídas del ECG.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Funciones auxiliares para métricas personalizadas
def sensitivity_score(y_true, y_pred):
    """
    Calcula la sensibilidad (recall para la clase positiva).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def specificity_score(y_true, y_pred):
    """
    Calcula la especificidad (tasa de verdaderos negativos).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


class ECGClassifier:
    """
    Clase para entrenar y usar modelos de clasificación de ECG.
    """
    
    def __init__(self, model_type='svm'):
        """
        Inicializa el clasificador.
        
        Args:
            model_type (str): 'svm' o 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.classes = ['Normal', 'Arrhythmia', 'Apnea']
    
    def _create_model(self):
        """Crea la instancia del modelo según el tipo."""
        if self.model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError("model_type debe ser 'svm' o 'random_forest'")
    
    def generate_synthetic_data(self, n_samples=300):
        """
        Genera datos sintéticos de entrenamiento para demostración.
        
        Args:
            n_samples (int): Número total de muestras a generar.
        
        Returns:
            tuple: (X, y) Features y labels.
        """
        n_per_class = n_samples // 3
        
        # Clase 1: Normal
        normal_samples = np.random.randn(n_per_class, 6) * [20, 5, 100, 2, 50, 10] + \
                        [70, 15, 600, 0.2, 200, 100]  # HR, RMSSD, SDNN, apnea_idx, LF, HF
        normal_y = np.zeros(n_per_class)
        
        # Clase 2: Arrhythmia
        arrhythmia_samples = np.random.randn(n_per_class, 6) * [25, 10, 120, 3, 60, 20] + \
                            [100, 25, 500, 0.5, 300, 150]
        arrhythmia_y = np.ones(n_per_class)
        
        # Clase 3: Apnea
        apnea_samples = np.random.randn(n_per_class, 6) * [30, 15, 150, 5, 70, 25] + \
                       [60, 30, 700, 2.0, 250, 80]
        apnea_y = np.ones(n_per_class) * 2
        
        X = np.vstack([normal_samples, arrhythmia_samples, apnea_samples])
        y = np.hstack([normal_y, arrhythmia_y, apnea_y])
        
        # Mezclar aleatoriamente
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        return X, y
    
    def train(self, X, y):
        """
        Entrena el modelo.
        
        Args:
            X (array): Matriz de características (n_samples, n_features).
            y (array): Vector de etiquetas.
        """
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Crear y entrenar modelo
        self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Calcular métricas
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Para métricas binarias, usar one-vs-rest
        sensitivity = []
        specificity = []
        for class_idx in range(len(self.classes)):
            y_test_binary = (y_test == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            tp = np.sum((y_test_binary == 1) & (y_pred_binary == 1))
            tn = np.sum((y_test_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_test_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_test_binary == 1) & (y_pred_binary == 0))
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivity.append(sens)
            specificity.append(spec)
        
        self.is_fitted = True
        self.metrics = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
        
        return self.metrics
    
    def predict(self, features):
        """
        Predice la clase para nuevas características.
        
        Args:
            features (array): Array con 6 características: 
                [HR, RMSSD, SDNN, apnea_index, LF, HF]
        
        Returns:
            dict: Predicción y confianza.
        """
        if not self.is_fitted:
            return {
                'class': 'Unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Asegurar que features es 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        predicted_class = self.classes[int(prediction)]
        
        # Obtener probabilidades
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
        else:
            # Para SVM sin probabilidades calibradas
            probabilities = np.ones(len(self.classes)) / len(self.classes)
        
        confidence = probabilities[int(prediction)]
        
        prob_dict = {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'probabilities': prob_dict
        }


def create_feature_vector(features_dict):
    """
    Crea un vector de características para clasificación.
    
    Args:
        features_dict (dict): Diccionario retornado por extract_all_features.
    
    Returns:
        array: Vector de 6 características normalizadas.
    """
    hr = features_dict['heart_rate']
    rmssd = features_dict['hrv_time_domain']['RMSSD']
    sdnn = features_dict['hrv_time_domain']['SDNN']
    apnea_idx = features_dict['apnea_features']['apnea_index']
    lf = features_dict['hrv_frequency_domain']['LF']
    hf = features_dict['hrv_frequency_domain']['HF']
    
    # Normalizar a rangos razonables
    feature_vector = np.array([
        np.clip(hr, 30, 200),
        np.clip(rmssd, 0, 200),
        np.clip(sdnn, 0, 200),
        np.clip(apnea_idx, 0, 100),
        np.clip(lf, 0, 1000),
        np.clip(hf, 0, 1000)
    ])
    
    return feature_vector


def train_classifiers():
    """
    Entrena dos clasificadores (SVM y Random Forest) con datos sintéticos.
    
    Returns:
        tuple: (svm_classifier, rf_classifier)
    """
    # Generar datos de entrenamiento
    X, y = ECGClassifier().generate_synthetic_data(n_samples=300)
    
    # Entrenar SVM
    svm_clf = ECGClassifier(model_type='svm')
    svm_metrics = svm_clf.train(X, y)
    
    # Entrenar Random Forest
    rf_clf = ECGClassifier(model_type='random_forest')
    rf_metrics = rf_clf.train(X, y)
    
    return svm_clf, rf_clf, svm_metrics, rf_metrics
