import os
import chromadb
import truststore

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# בדיקת תעודות HTTPS דרך Windows (שם מותקנת תעודת הסינון). הבדיקה לא מכובה
truststore.inject_into_ssl()

# טוען את המשתנים מקובץ .env אל משתני הסביבה
load_dotenv()

# הגדרות
DOCS_PATH = "./docs"
# הגודל נמדד ב-tokens. עברית "יקרה" ב-tokens, לכן 512 ולא 250 כמו אצל המורה
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "gemini-embedding-001"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"


def setup_embedding():
    """הגדרת מודל ה-Embeddings (כל קטע יהפוך לוקטור של מספרים)."""
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=os.getenv("GEMINI_API_KEY"),
    )


def load_documents():
    """טעינת כל קובצי ה-Markdown מתיקיית docs (כל קובץ = Document אחד)."""
    return SimpleDirectoryReader(
        DOCS_PATH,
        required_exts=[".md"],
    ).load_data()


def split_documents(documents):
    """חלוקה לקטעים: קודם לפי כותרות, ואז חיתוך של קטעים ארוכים מדי."""
    # שלב א: קטע אחד לכל כותרת Markdown (#, ##)
    markdown_parser = MarkdownNodeParser()
    nodes = markdown_parser.get_nodes_from_documents(documents)

    # שלב ב: קטע שעובר את CHUNK_SIZE נחתך לכמה קטעים עם חפיפה
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(nodes)

    # מדלגים על קטע שיש בו רק כותרת בלי תוכן (למשל "# מה זה BabyLink")
    return [node for node in nodes if "\n" in node.get_content().strip()]


def print_chunks(nodes):
    """הדפסת כל הקטעים: שם הקובץ, הכותרת והטקסט."""
    for i, node in enumerate(nodes, start=1):
        print(f"===== Chunk {i} =====")
        print(f"File:   {node.metadata.get('file_name')}")
        print(f"Header: {node.metadata.get('header_path')}")
        print(node.get_content())
        print()

    print(f"Total chunks: {len(nodes)}")


def setup_vector_store():
    """חיבור ל-Chroma. מוחקים את ה-collection הישן ובונים חדש בכל הרצה."""
    client = chromadb.PersistentClient(path=DB_PATH)

    # מחיקה, כדי שקטעים ממסמך שהשתנה או נמחק לא יישארו במאגר
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    # cosine = מודדים דמיון לפי הזווית בין הוקטורים
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    store = ChromaVectorStore(chroma_collection=collection)
    return StorageContext.from_defaults(vector_store=store), collection


def save_chunks(nodes, storage_context):
    """יצירת Embedding לכל קטע ושמירה ב-Chroma."""
    # כאן נשלחים הקטעים ל-Gemini, וכל קטע חוזר כוקטור
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    return index


def main():
    """הפונקציה הראשית שמפעילה את התהליך."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Missing GEMINI_API_KEY")

    setup_embedding()

    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    nodes = split_documents(documents)
    # print_chunks(nodes)  # להדפסת כל הקטעים (כמו בשלב 1)

    storage_context, collection = setup_vector_store()

    save_chunks(nodes, storage_context)

    print(f"Total chunks in ChromaDB: {collection.count()}")


if __name__ == "__main__":
    main()
