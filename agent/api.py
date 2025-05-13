# api.py
from fastapi import FastAPI, Request
from main import temizle, vector_db, is_ml_related, get_gemini_answer
from langchain_core.documents import Document

app = FastAPI()

@app.post("/ask")
async def ask_question(req: Request):
    data = await req.json()
    user_input = data.get("question", "").strip()
    if not user_input:
        return {"answer": "Soru eksik."}

    temiz_soru = temizle(user_input)
    sonuclar = vector_db.similarity_search_with_score(temiz_soru, k=3)

    en_iyi_sonuc = None
    en_iyi_skor = 1.0

    for doc, skor in sonuclar:
        if skor < en_iyi_skor:
            en_iyi_sonuc = doc
            en_iyi_skor = skor

    if en_iyi_skor < 0.8:
        return {
            "answer": en_iyi_sonuc.metadata["cevap"],
            "source": en_iyi_sonuc.metadata.get("source", "unknown")
        }

    if not is_ml_related(user_input):
        return {"answer": "Bu soru makine öğrenmesiyle ilgili değil."}

    yanit = get_gemini_answer(user_input)
    if "Gemini API'den yanıt alınamadı" in yanit:
        return {"answer": "API'den yanıt alınamadı."}

    yeni_doc = Document(page_content=temiz_soru, metadata={"cevap": yanit, "source": "gemini"})
    vector_db.add_documents([yeni_doc])
    vector_db.persist()
    
    return {"answer": yanit, "source": "gemini"}
