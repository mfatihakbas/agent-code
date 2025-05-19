from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from airtable_client import temizle, embedding_model, add_record, bul_benzer_kayit, increment_usage_count
from openai_client import get_openai_answer
from utils import is_ml_related
import time

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    user_input = req.question.strip()

    if not is_ml_related(user_input):
        return JSONResponse(content={"reply_answer": "⚠️ Bu soru makine öğrenmesiyle ilgili değil."})

    temiz_soru = temizle(user_input)
    embed = embedding_model.encode([temiz_soru])[0].tolist()

    start = time.time()
    kayit, skor, record_id = bul_benzer_kayit(embed)
    elapsed = round(time.time() - start, 4)

    if kayit:
        increment_usage_count(record_id)
        mesaj = f"🧠 Yanıt: {kayit['answer']}"
        return JSONResponse(content={"reply_answer": mesaj})

    yanit = get_openai_answer(user_input)

    if yanit and "OpenAI API'den yanıt alınamadı" not in yanit:
        add_record(user_input, yanit, source="openai", embedding=embed)
        mesaj = f"🧠 Yanıt: {yanit}"
        return JSONResponse(content={"reply_answer": mesaj})

    return JSONResponse(content={"reply_answer": "❌ Şu anda yanıt üretilemedi."})
