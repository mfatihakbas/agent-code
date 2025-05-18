import os
import numpy as np
import re
from dotenv import load_dotenv
from pyairtable import Table
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# .env'den yükle
load_dotenv()
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

table = Table(AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 🔹 Soru temizleyici
def temizle(metin):
    stop_words = {"nedir", "ne", "nasıl", "ile", "ve", "için", "demektir", "?"}
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    kelimeler = metin.split()
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
    return ' '.join(temiz_kelimeler).strip()

# ➕ Yeni kayıt ekleme
def add_record(question: str, answer: str, source: str = "manual", embedding: list = None):
    record = {
        "question": question,
        "answer": answer,
        "source": source,
        "usage_count": 1
    }
    if embedding:
        record["embedding"] = str(embedding)
    try:
        table.create(record)
        print("✅ Airtable’a kayıt eklendi.")
    except Exception as e:
        print("❌ Airtable ekleme hatası:", e)

def increment_usage_count(record_id):
    try:
        # Mevcut kaydı getir
        current_record = table.get(record_id)
        current_count = current_record["fields"].get("usage_count", 0)

        # Sayacı +1 yap
        new_count = int(current_count) + 1

        # Güncelle
        table.update(record_id, {"usage_count": new_count})
        print(f"🔄 usage_count güncellendi: {new_count}")
    except Exception as e:
        print("❌ Sayaç güncelleme hatası:", e)

# 🔍 En iyi eşleşmeyi usage_count öncelikli bul
def bul_benzer_kayit(embed, threshold=0.90):
    records = table.all()

    # 🔽 usage_count >= 1 ve sıralı
    aktif_kayitlar = [
        r for r in records
        if "embedding" in r["fields"] and r["fields"].get("usage_count", 0) >= 1
    ]
    aktif_kayitlar.sort(key=lambda r: r["fields"].get("usage_count", 0), reverse=True)

    en_iyi_kayit = None
    en_iyi_skor = 0
    en_iyi_id = None

    for r in aktif_kayitlar:
        fields = r["fields"]
        try:
            kayit_embed = np.array(eval(fields["embedding"]))
            skor = cosine_similarity([embed], [kayit_embed])[0][0]
            if skor > en_iyi_skor and skor >= threshold:
                en_iyi_skor = skor
                en_iyi_kayit = fields
                en_iyi_id = r["id"]
        except Exception:
            continue

    return en_iyi_kayit, en_iyi_skor, en_iyi_id

# 📥 Örnek verileri ilk kez yükle
def yukle_ornek_veriler():
    print("📥 Airtable'a örnek veriler yükleniyor...")
    orijinal_sorular = [
        ("Makine öğrenmesi nedir?", "Makine öğrenmesi, veriden öğrenen ve tahmin yapabilen algoritmalardır."),
        ("Denetimli öğrenme ne demektir?", "Denetimli öğrenme, etiketli veriyle modeli eğitmektir."),
        ("Denetimsiz öğrenme nedir?", "Denetimsiz öğrenme, verideki yapıları etiket olmadan keşfetmektir."),
        ("Veri kümesi nedir?", "Veri kümesi, modelin öğrenmesi için kullanılan örnekler topluluğudur."),
        ("Model nedir?", "Model, öğrenilen bilgiyi temsil eden matematiksel yapıdır."),
        ("Overfitting nedir?", "Overfitting, modelin eğitildiği veriye aşırı uyum sağlamasıdır."),
        ("Aşırı öğrenme nedir?", "Overfitting, modelin eğitildiği veriye aşırı uyum sağlamasıdır."),
        ("Accuracy nasıl hesaplanır?", "Doğruluk, doğru tahminlerin toplam tahminlere oranıdır."),
        ("Doğruluk hesaplama nedir?", "Doğruluk, doğru tahminlerin toplam tahminlere oranıdır."),
        ("Eğitim ve test verisi neden ayrılır?", "Eğitim verisi modelin öğrenmesi, test verisi performans değerlendirmesi içindir."),
        ("Lineer regresyon nedir?", "Lineer regresyon, doğrusal ilişki kurarak tahmin yapan bir yöntemdir."),
        ("Lineer regresyon ne işe yarar?", "Lineer regresyon, doğrusal ilişki kurarak tahmin yapan bir yöntemdir."),
        ("Scikit-learn ne işe yarar?", "Scikit-learn, Python'da makine öğrenmesi modelleri geliştirmek için kullanılır.")
    ]

    mevcut_sorular = {r["fields"].get("question") for r in table.all() if "question" in r["fields"]}

    for soru, cevap in orijinal_sorular:
        if soru in mevcut_sorular:
            continue
        temiz_soru = temizle(soru)
        embedding = embedding_model.encode([temiz_soru])[0].tolist()
        add_record(soru, cevap, source="manual", embedding=embedding)

    print("✅ Örnek veriler yüklendi.\n")
