# agent/openai_client.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# .env yükle
load_dotenv()

# Client başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Soru-cevap fonksiyonu
def get_openai_answer(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Türkçe teknik sorulara yardımcı olan bir asistansın."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ OpenAI API hatası:\n", e)
        return "OpenAI API'den yanıt alınamadı."
