from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

# הגדרות
DOCS_PATH = "./docs"
# הגודל נמדד ב-tokens. עברית "יקרה" ב-tokens, לכן 512 ולא 250 כמו אצל המורה
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


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


def main():
    """הפונקציה הראשית שמפעילה את התהליך."""
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    nodes = split_documents(documents)
    print_chunks(nodes)


if __name__ == "__main__":
    main()
