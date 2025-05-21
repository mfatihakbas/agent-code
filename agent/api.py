from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time

from airtable_client import (
    temizle,
    embedding_model,
    add_record,
    bul_benzer_kayit,
    increment_usage_count,
    update_secondary_answer,
    get_record_by_id
)

from openai_client import get_openai_answer
from utils import is_ml_related

app = FastAPI()

# 🟦 Model tanımları
class QuestionRequest(BaseModel):
    question: str

class RetryRequest(BaseModel):
    question: str
    record_id: str


# 🔹 İlk soru sorma endpoint'i
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    user_input = req.question.strip()

    # ❌ Makine öğrenmesi dışındaki sorular engellenir
    if not is_ml_related(user_input):
        return JSONResponse(content={"reply_answer": "⚠️ Bu soru makine öğrenmesiyle ilgili değil."})

    # 🧠 Embedding hesaplanır
    temiz_soru = temizle(user_input)
    embed = embedding_model.encode([temiz_soru])[0].tolist()

    # 🔍 En benzer kayıt aranır
    start = time.time()
    kayit, skor, record_id = bul_benzer_kayit(embed)
    elapsed = round(time.time() - start, 4)

    if kayit:
        increment_usage_count(record_id)
        mesaj = f"🧠 Yanıt: {kayit['answer']}"
        return JSONResponse(content={
            "reply_answer": mesaj,
            "record_id": record_id
        })

    # 🤖 OpenAI'den yeni cevap alınır
    yanit = get_openai_answer(user_input)
    if yanit and "OpenAI API'den yanıt alınamadı" not in yanit:
        add_record(user_input, yanit, source="openai", embedding=embed)
        mesaj = f"🧠 Yanıt: {yanit}"
        return JSONResponse(content={"reply_answer": mesaj})

    return JSONResponse(content={"reply_answer": "❌ Şu anda yanıt üretilemedi."})


# 🔁 Kullanıcı cevabı beğenmediğinde alternatif üretir
@app.post("/retry_answer")
async def retry_answer(req: RetryRequest):
    question = req.question.strip()
    record_id = req.record_id.strip()

    # 📄 Airtable'dan kayıt alınır
    fields = get_record_by_id(record_id)
    if fields is None:
        return JSONResponse(content={"reply_answer": "❌ Kayıt bulunamadı."})

    # 🔄 Daha önce üretilmiş alternatif varsa onu göster
    if fields.get("secondary_answer"):
        return JSONResponse(content={"reply_answer": f"🔁 Alternatif Cevap: {fields['secondary_answer']}"})

    # 🤖 Yeni alternatif cevap oluştur
    new_answer = get_openai_answer(question)
    if not new_answer or "OpenAI API'den yanıt alınamadı" in new_answer:
        return JSONResponse(content={"reply_answer": "❌ Yeni yanıt üretilemedi."})

    # 💾 Alternatif cevap Airtable'a kaydedilir
    update_secondary_answer(record_id, new_answer)
    return JSONResponse(content={"reply_answer": f"🤖 Yeni Cevap: {new_answer}"})
