# utils.py

ml_keywords = [
    # Temel kavramlar
    "makine öğrenmesi", "machine learning", "yapay zeka", "veri kümesi", 
    "eğitim verisi", "test verisi", "label", "feature", "etiket", "özellik", "target", "input", "output",
    
    # Öğrenme türleri
    "denetimli", "denetimsiz", "yarı denetimli", "pekiştirmeli öğrenme", 
    "reinforcement learning", "semi-supervised", "unsupervised learning", "supervised learning",

    # Doğruluk / değerlendirme
    "doğruluk", "accuracy", "precision", "recall", "f1 skoru", "confusion matrix", 
    "ROC", "AUC", "değerlendirme metriği", "performans metriği", "loss", "hata oranı",

    # Problemler
    "overfitting", "underfitting", "bias", "variance", "genelleme", "aşırı öğrenme",

    # Model türleri ve algoritmalar
    "regresyon", "sınıflandırma", "clustering", "kümeleme", 
    "karar ağacı", "decision tree", "random forest", "logistic regression", "linear regression", 
    "naive bayes", "k-nearest neighbors", "knn", "svm", "support vector machine", 
    "ensemble", "bagging", "boosting", "xgboost", "lightgbm", "gradient boosting", "catboost",

    # Derin öğrenme
    "deep learning", "derin öğrenme", "yapay sinir ağı", "neural network", 
    "cnn", "convolutional neural network", "rnn", "lstm", "transformer", 
    "attention", "self-attention", "seq2seq", "autoencoder", "gan", "generative adversarial network",

    # Özellik mühendisliği
    "feature engineering", "feature extraction", "feature selection", 
    "pca", "principal component analysis", "özellik seçimi", "boyut indirgeme",

    # Veri ön işleme
    "feature scaling", "normalizasyon", "standartlaştırma", "min-max scaling", 
    "z-score", "eksik veri", "missing value", "outlier", "kategori kodlama", 
    "one-hot encoding", "label encoding", "veri temizleme", "ön işleme", "data preprocessing",

    # Eğitim yöntemleri
    "cross validation", "k-fold", "early stopping", "grid search", "random search", 
    "hiperparametre", "hyperparameter tuning", "model seçimi", "validation set",

    # Kütüphaneler ve framework'ler
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "lightgbm", 
    "statsmodels", "mlflow", "huggingface", "optuna", "transformers",

    # Uygulama alanları
    "doğal dil işleme", "nlu", "nlp", "bilgisayarla görme", "görüntü işleme", 
    "öneri sistemi", "recommendation system", "zaman serisi", "time series", 
    "anomalie tespiti", "anomaly detection", "makine arızası tahmini",

    # Diğer önemli kavramlar
    "pipeline", "model pipeline", "skor", "metriks", "veri bölme", "train test split", 
    "data augmentation", "tokenization", "embedding", "word2vec", "bert", "gpt",
    "transfer öğrenme", "transfer learning", "veri dengesizliği", "class imbalance", "sampling", 
    "oversampling", "undersampling", "smote"
]


def is_ml_related(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in ml_keywords)
