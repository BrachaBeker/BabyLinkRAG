# BabyLink RAG – עוזר הצ'אט

עוזר שעונה על שאלות על [BabyLink](https://babylinkclient.onrender.com) (איך מפרסמים בקשה, מדיניות ביטולים, פרטיות, תמיכה) **רק לפי מסמכי הידע** שב-`docs/`. בנוי כמערכת **RAG** עם LlamaIndex, Gemini ו-ChromaDB. מה שאין עליו מידע מופנה לתמיכה.

🔗 [לקוח](https://babylinkclient.onrender.com) · [Repository הלקוח](https://github.com/BrachaBeker/BabyLinkClient) (בועת הצ'אט: `src/components/ChatWidget.tsx`)

## ארכיטקטורה
```
דפדפן (React, בועת צ'אט)  --POST /api/chat-->  api.py (FastAPI)  -->  rag.py  -->  Gemini
                                                                        |
                                                                    ChromaDB (chroma_db/)
```
| קובץ | תפקיד |
|---|---|
| `ingest.py` | טוען את `docs/`, מחלק לקטעים (לפי כותרות Markdown, ואז `SentenceSplitter`), יוצר Embedding לכל קטע ושומר ב-Chroma. בונה את המאגר מחדש בכל הרצה |
| `rag.py` | הלב: `retrieve_documents` (שליפת `TOP_K` קטעים) ו-`generate_answer` (תשובה לפי המקורות בלבד, `temperature=0`) |
| `api.py` | `POST /api/chat` (`{question, history}` ← `{answer, sources}`), `GET /health`, CORS, ולידציה ושגיאת 503 כשהמודל לא זמין |
| `search.py` · `ask.py` · `eval_questions.py` | בדיקות בטרמינל: שליפה בלבד, צ'אט, ו-Recall@k על שאלות בדיקה |

## החלטות
- **מודלים:** Embeddings `gemini-embedding-001`. תשובות: `GEMINI_CHAT_MODEL` (ברירת מחדל `gemini-3.5-flash`), ואם נגמרה לו המכסה, מודל גיבוי (`gemini-3.1-flash-lite`). שאלת המשך ("ומה לגבי...?") מנוסחת מחדש כשאלה עצמאית לפני החיפוש, במודל קל (`gemini-3.5-flash-lite`). לכל מודל מכסה חינמית נפרדת.
- **Chunks:** `CHUNK_SIZE=512` tokens, כי עברית "יקרה" ב-tokens. כמעט כל סעיף במסמכים נשאר קטע שלם.
- **בטיחות:** תוכן המסמכים הוא מידע ולא הוראות, היסטוריה רק עם `user`/`assistant`, שאלה עד 500 תווים.
- **השרת לא שומר מצב:** הבועה שולחת את 6 ההודעות האחרונות בכל בקשה.
- **`truststore`:** בדיקת תעודות HTTPS דרך מערכת ההפעלה (נדרש מאחורי סינון אינטרנט). הבדיקה לא מכובה.

## הרצה מקומית
דרישות: Python 3.13 ומפתח Gemini מ-[Google AI Studio](https://aistudio.google.com/apikey).
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env               # למלא GEMINI_API_KEY
python check_key.py                  # בדיקה שהמפתח עובד
python ingest.py                     # בניית המאגר
uvicorn api:app --reload --port 8000 # http://localhost:8000/docs
```

## עדכון מסמך
עורכים או מוסיפים קובץ Markdown ב-`docs/` ומריצים `python ingest.py`. ב-Render המאגר נבנה מחדש בכל Deploy.

## פריסה (Render Web Service)
- Build: `pip install -r requirements.txt && python ingest.py` · Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Environment: `GEMINI_API_KEY`, `GEMINI_CHAT_MODEL`, `ALLOWED_ORIGINS=https://babylinkclient.onrender.com` (בלי `/` בסוף). גרסת Python מ-`.python-version`.
- בלקוח: `VITE_CHAT_URL` עם כתובת השירות, ו-Deploy מחדש.
