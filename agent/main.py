import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from gemini_client import get_gemini_answer
from utils import is_ml_related
from airtable_client import add_record


# 🔹 Metni normalize eden fonksiyon (küçük harf, noktalama temizliği vs.)
def temizle(metin):
    stop_words = {"nedir", "ne", "nasıl", "ile", "ve", "için", "demektir", "?"}
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin)  # noktalama kaldır
    kelimeler = metin.split()
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
    return ' '.join(temiz_kelimeler).strip()

# 🔹 Türkçe destekli güçlü embedding modeli
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 🔹 ChromaDB başlat
vector_db = Chroma(
    collection_name="soru_cevaplar",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

# 🔹 Eğer veritabanı boşsa örnek verileri ekle (temizlenmiş haliyle!)
if vector_db._collection.count() == 0:
    print("📥 İlk örnek veri seti yükleniyor...")

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

    sorular = [
        Document(page_content=temizle(soru), metadata={"cevap": cevap, "source": "manual"})
        for soru, cevap in orijinal_sorular
    ]

    vector_db.add_documents(sorular)
    vector_db.persist()
    print(f"✅ {len(sorular)} örnek soru başarıyla eklendi.\n")

# 🔁 Kullanıcı ile etkileşim
print("\n📌 ML Soru-Cevap Asistanına Hoş Geldiniz. Çıkmak için 'q' yazın.\n")

while True:
    user_input = input("❓ Sorunuzu yazın: ").strip()

    if user_input.lower() in ["q", "çık", "exit"]:
        print("👋 Görüşmek üzere!")
        break

    temiz_soru = temizle(user_input)
    sonuclar = vector_db.similarity_search_with_score(temiz_soru, k=3)

    en_iyi_sonuc = None
    en_iyi_skor = 1.0

    for doc, skor in sonuclar:
        if skor < en_iyi_skor:
            en_iyi_sonuc = doc
            en_iyi_skor = skor

    if en_iyi_skor < 0.8:
        source = en_iyi_sonuc.metadata.get("source", "unknown")
        if source == "manual":
            kaynak_bilgi = "📚 Örnek veri setinden"
        elif source == "gemini":
            kaynak_bilgi = "🤖 Gemini API'den"
        else:
            kaynak_bilgi = "📥 Sonradan eklenmiş"

        print(f"🧠 Cevap ({kaynak_bilgi}):\n{en_iyi_sonuc.metadata['cevap']}")
        continue


    # 💡 Veritabanında yoksa, ML ile ilgili mi kontrol et
    if not is_ml_related(user_input):
        print("⚠️ Bu soru makine öğrenmesiyle ilgili görünmüyor, bu yüzden cevap verilmeyecek.")
        continue

    # 🌐 Gemini API'ye sor
    print("🤖 Gemini API'ye soruluyor...")
    yanit = get_gemini_answer(user_input)

    if yanit and "Gemini API'den yanıt alınamadı" not in yanit:
        print(f"🧠 Cevap (🤖 Gemini API'den):\n{yanit}")

        yeni_doc = Document(page_content=temiz_soru, metadata={"cevap": yanit, "source": "gemini"})
        vector_db.add_documents([yeni_doc])
        vector_db.persist()
        print("✅ Yeni veri ChromaDB’ye eklendi.")

        # 🔗 Airtable’a da ekle
        add_record(user_input, yanit, source="gemini")
    else:
        print("❌ Gemini'den yanıt alınamadı.")
