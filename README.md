# 🫀 CardioLab – Detection of Arrhythmias and Sleep Apnea from ECG Signals

CardioLab is a biomedical signal analysis tool that processes ECG signals to automatically detect arrhythmias and sleep apnea using digital signal processing and machine learning models (SVM, Random Forest, CNN).

## 🚀 Features
- ECG signal preprocessing (noise removal, filtering)
- HRV and RR interval analysis
- Classification of arrhythmia and apnea events
- Synthetic ECG signal generation for model validation
- Streamlit web interface

## 🧰 Tech Stack
- Python 3.11
- Streamlit 1.38
- NumPy, Pandas, Scikit-learn, Matplotlib

## ▶️ Run Locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
