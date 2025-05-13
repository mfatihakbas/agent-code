import os
from dotenv import load_dotenv
from pyairtable import Table

# .env dosyasını yükle
load_dotenv()

# Gerekli bilgileri al
api_key = os.getenv("AIRTABLE_API_KEY")
base_id = os.getenv("AIRTABLE_BASE_ID")
table_name = os.getenv("AIRTABLE_TABLE_NAME")

# Airtable tablosuna eriş
table = Table(api_key, base_id, table_name)

# 🔹 Yeni kayıt ekleme fonksiyonu
def add_record(question: str, answer: str, source: str = "manual"):
    record = {
        "question": question,
        "answer": answer,
        "source": source
    }
    try:
        table.create(record)
        print("✅ Airtable’a kayıt eklendi.")
    except Exception as e:
        print("❌ Airtable ekleme hatası:", e)

# 🔹 Tüm kayıtları listeleme (isteğe bağlı)
def list_records():
    try:
        records = table.all()
        for r in records:
            print(f"📝 {r['fields'].get('question')} → {r['fields'].get('answer')} ({r['fields'].get('source')})")
    except Exception as e:
        print("❌ Airtable okuma hatası:", e)
# 🔍 Bağımsız test için:
if __name__ == "__main__":
    # Airtable'a örnek kayıt ekle
    add_record(
        question="Makine öğrenmesi nedir?",
        answer="Veriden öğrenip tahmin yapan algoritmalardır.",
        source="test"
    )

    # Airtable'daki tüm kayıtları listele
    print("\n📋 Airtable'daki kayıtlar:")
    list_records()
