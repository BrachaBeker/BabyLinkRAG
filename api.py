import os

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag import (
    get_clients,
    build_search_queries,
    retrieve_for_queries,
    generate_answer,
    MAX_HISTORY_MESSAGES,
)

# הגדרות
MAX_QUESTION_LENGTH = 500
MAX_MESSAGE_LENGTH = 2000

# הכתובות (Origins) שמותר להן לפנות לשרת מהדפדפן, מופרדות בפסיק
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

# כאן נשמרים המודל והאינדקס אחרי שהשרת עולה
clients = {}


# ==========================================
# Request / Response models (מבנה ה-JSON)
# ==========================================

class HistoryMessage(BaseModel):
    """הודעה אחת מההיסטוריה של השיחה."""
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    """מה שהלקוח שולח: { question, history }."""
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    history: list[HistoryMessage] = Field(default=[], max_length=MAX_HISTORY_MESSAGES)


class ChatResponse(BaseModel):
    """מה שהשרת מחזיר: { answer, sources }."""
    answer: str
    sources: list[str]


# ==========================================
# App
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """רץ פעם אחת כשהשרת עולה: טוען את המודלים ואת האינדקס (ולא בכל שאלה)."""
    llms, rewrite_llm, index, collection = get_clients()
    clients["llms"] = llms
    clients["rewrite_llm"] = rewrite_llm
    clients["index"] = index
    print(f"Chunks in database: {collection.count()}")
    yield


app = FastAPI(title="BabyLink Chat", lifespan=lifespan)

# CORS: מאפשר לדפדפן באתר ה-React לקרוא לשרת הזה, למרות שהוא בכתובת אחרת
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    """בדיקה שהשרת חי."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """מקבל שאלה והיסטוריה, ומחזיר תשובה ומקורות (דרך rag.py)."""
    # rag.py עובד עם מילונים רגילים, לא עם מודלים של Pydantic
    history = [message.model_dump() for message in request.history]

    try:
        # שאלת המשך הופכת לשאלה עצמאית, ושאלה כפולה מתפרקת - ולכל שאלה חיפוש משלה
        queries = build_search_queries(
            request.question, history, clients["rewrite_llm"]
        )
        documents = retrieve_for_queries(queries, clients["index"])
        answer, sources = generate_answer(
            request.question, documents, clients["llms"], history, queries
        )
    except Exception as error:
        # Gemini לא זמין / חרגנו מהמכסה - שגיאה מסודרת ב-JSON במקום קריסה
        print(f"Chat error: {error}")
        return JSONResponse(
            status_code=503,
            content={"detail": "The assistant is not available right now. Try again later."},
        )

    return ChatResponse(answer=answer, sources=sources)
