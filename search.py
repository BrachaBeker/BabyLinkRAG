import os
import chromadb
import truststore

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# בדיקת תעודות HTTPS דרך Windows (שם מותקנת תעודת הסינון). הבדיקה לא מכובה
truststore.inject_into_ssl()

# טוען את המשתנים מקובץ .env אל משתני הסביבה
load_dotenv()

# הגדרות (חייבות להיות זהות ל-ingest.py, אחרת לא נמצא את הקטעים)
EMBEDDING_MODEL = "gemini-embedding-001"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"

# כמה קטעים מחזירים לכל שאלה
TOP_K = 4


def get_index():
    """חיבור ל-Chroma הקיים (שנבנה ב-ingest.py) והחזרת אינדקס לחיפוש."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    # אותו מודל Embeddings כמו ב-ingest.py, כדי שהשאלה והקטעים יהיו באותו "מרחב"
    embed_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=api_key,
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        raise RuntimeError("The database is empty. Run: python ingest.py")

    vector_store = ChromaVectorStore(chroma_collection=collection)

    # from_vector_store - לא יוצרים Embeddings מחדש, רק מתחברים למה שכבר שמור
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def retrieve_documents(question, index):
    """מחזיר את TOP_K הקטעים הכי קרובים לשאלה, עם הציון ושם הקובץ."""
    retriever = index.as_retriever(similarity_top_k=TOP_K)

    # כאן השאלה הופכת ל-Embedding ו-Chroma מחפש את הוקטורים הכי קרובים
    results = retriever.retrieve(question)
    documents = []

    for result in results:
        node = result.node
        text = node.get_content().strip()

        if not text:
            continue

        metadata = node.metadata or {}

        documents.append({
            "text": text,
            "source": metadata.get("file_name", "Unknown source"),
            "score": result.score,
        })

    return documents


def main():
    """שאלות בטרמינל: לכל שאלה מדפיסים את הקטעים שנמצאו. Enter ריק = יציאה."""
    index = get_index()

    while True:
        question = input("\nשאלה (Enter ליציאה): ").strip()
        if not question:
            break

        documents = retrieve_documents(question, index)

        for i, doc in enumerate(documents, start=1):
            print(f"\n----- {i}. score: {doc['score']:.3f} | {doc['source']} -----")
            print(doc["text"])


if __name__ == "__main__":
    main()
