# utils.py

ml_keywords = [
    "makine öğrenmesi", "machine learning", "denetimli", "denetimsiz",
    "veri kümesi", "overfitting", "regresyon", "sınıflandırma",
    "doğruluk", "cross validation", "scikit-learn", "tensorflow",
    "model", "eğitim verisi", "test verisi", "feature", "label",
    "yapay zeka", "derin öğrenme", "neural network"
]

def is_ml_related(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in ml_keywords)
