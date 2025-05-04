from sentence_transformers import SentenceTransformer
import pandas as pd

# 🔹 Basic machine learning Q&A dataset in Turkish
data = {
    "soru": [
        "Makine öğrenmesi nedir?",
        "Denetimli öğrenme ne demektir?",
        "Denetimsiz öğrenme nedir?",
        "Veri kümesi (dataset) nedir?",
        "Makine öğrenmesinde model ne demektir?",
        "Overfitting ne anlama gelir?",
        "Doğruluk (accuracy) nasıl hesaplanır?",
        "Makine öğrenmesinde eğitim ve test verisi neden ayrılır?",
        "Lineer regresyon ne işe yarar?",
        "Scikit-learn hangi amaçla kullanılır?"
    ],
    "cevap": [
        "Makine öğrenmesi, veriden öğrenen ve tahmin yapabilen algoritmalardır.",
        "Denetimli öğrenme, etiketli veriyle modeli eğitmektir.",
        "Denetimsiz öğrenme, verideki yapıları etiket olmadan keşfetmektir.",
        "Veri kümesi, modelin öğrenmesi için kullanılan örnekler topluluğudur.",
        "Model, öğrenilen bilgiyi temsil eden matematiksel yapıdır.",
        "Overfitting, modelin eğitildiği veriye aşırı uyum sağlayarak genelleme yapamamasıdır.",
        "Doğruluk, doğru tahminlerin toplam tahminlere oranıdır.",
        "Eğitim verisi öğrenmek, test verisi değerlendirmek için kullanılır.",
        "Lineer regresyon, sürekli değişkenler arasında ilişki kurar ve tahmin yapar.",
        "Scikit-learn, Python ile makine öğrenmesi algoritmalarını uygulamak için kullanılır."
    ]
}

# 🔹 Create a DataFrame from the data
df = pd.DataFrame(data)

# 🔹 Load the multilingual embedding model (supports Turkish)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 🔹 Generate embeddings for each question
df["embedding"] = df["soru"].apply(lambda x: model.encode(x))

# 🔹 Print the embedding of the first question for verification
print("Sample embedding:")
print(df["embedding"].iloc[0])

# 🔹 [Optional] Save the DataFrame with embeddings to a file
df.to_pickle("week1/soru_cevap_embed.pkl")