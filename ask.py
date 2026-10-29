from rag import get_clients, rewrite_question, retrieve_documents, generate_answer

# True = מדפיס גם את השאלה שנשלחה לחיפוש, ואת הקטעים שנשלפו והציונים שלהם (לבדיקה)
SHOW_DOCUMENTS = False


def main():
    """צ'אט בטרמינל מעל rag.py. Enter ריק = יציאה."""
    llm, rewrite_llm, index, collection = get_clients()
    print(f"Chunks in database: {collection.count()}")

    # ההיסטוריה של השיחה הזו: [{"role": "user"/"assistant", "content": "..."}]
    history = []

    while True:
        question = input("\nשאלה (Enter ליציאה): ").strip()
        if not question:
            break

        # שאלת המשך מנוסחת מחדש כשאלה עצמאית, ורק איתה מחפשים
        search_question = rewrite_question(question, history, rewrite_llm)
        documents = retrieve_documents(search_question, index)

        if SHOW_DOCUMENTS:
            print(f"  חיפוש: {search_question}")
            for doc in documents:
                print(f"  [{doc['score']:.3f}] {doc['source']}")

        answer, sources = generate_answer(question, documents, llm, history)

        print(f"\nעוזר: {answer}")
        if sources:
            print(f"מקורות: {', '.join(sources)}")

        # שומרים את השאלה והתשובה, כדי ששאלת ההמשך תובן בהקשר
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
