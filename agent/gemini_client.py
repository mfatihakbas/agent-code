# gemini_client.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# API anahtarını al
api_key = os.getenv("GEMINI_API_KEY")

# Gemini API'yi yapılandır
genai.configure(api_key=api_key)

# Modeli başlat (gemini-pro metin modeli)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Soruya yanıt al
def get_gemini_answer(question: str) -> str:
    try:
        response = model.generate_content(question)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini API hatası:", e)
        return "Gemini API'den yanıt alınamadı."
