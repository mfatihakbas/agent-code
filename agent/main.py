import time  # ⏱ Süre ölçümü için
from openai_client import get_openai_answer
from utils import is_ml_related
from airtable_client import (
    temizle,
    embedding_model,
    add_record,
    bul_benzer_kayit,
    yukle_ornek_veriler,
    increment_usage_count
)

# 📥 Örnek verileri ilk çalıştırmada ekle
yukle_ornek_veriler()

print("\n📌 ML Soru-Cevap Asistanına Hoş Geldiniz. Çıkmak için 'q' yazın.\n")

while True:
    user_input = input("❓ Sorunuzu yazın: ").strip()

    if user_input.lower() in ["q", "çık", "exit"]:
        print("👋 Görüşmek üzere!")
        break

    if not is_ml_related(user_input):
        print("⚠️ Bu soru makine öğrenmesiyle ilgili görünmüyor.")
        continue

    temiz_soru = temizle(user_input)
    embed = embedding_model.encode([temiz_soru])[0].tolist()

    # ⏱️ Arama süresi başlat
    start = time.time()
    kayit, skor, record_id = bul_benzer_kayit(embed)
    elapsed = time.time() - start

    if kayit:
        print(f"\n🧠 Cevap (📄 Airtable'dan):\n{kayit['answer']}")
        print(f"📊 Benzerlik skoru: {skor:.4f}")
        print(f"⏱️ Arama süresi: {elapsed:.4f} saniye")
        increment_usage_count(record_id)
        continue

    print("🤖 OpenAI GPT'e soruluyor...")
    yanit = get_openai_answer(user_input)

    if yanit and "OpenAI API'den yanıt alınamadı" not in yanit:
        print(f"\n🧠 Cevap (🤖 OpenAI GPT'den):\n{yanit}")
        print(f"⏱️ Arama süresi (eşleşme bulunamadı): {elapsed:.4f} saniye")
        add_record(user_input, yanit, source="openai", embedding=embed)
    else:
        print("❌ OpenAI'den yanıt alınamadı.")
