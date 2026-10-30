from rag import get_clients, build_search_queries, retrieve_for_queries, generate_answer

# True = מדפיס גם את שאלות החיפוש, ואת הקטעים שנשלפו והציונים שלהם (לבדיקה)
SHOW_DOCUMENTS = False


def main():
    """צ'אט בטרמינל מעל rag.py. Enter ריק = יציאה."""
    llms, rewrite_llm, index, collection = get_clients()
    print(f"Chunks in database: {collection.count()}")

    # ההיסטוריה של השיחה הזו: [{"role": "user"/"assistant", "content": "..."}]
    history = []

    while True:
        question = input("\nשאלה (Enter ליציאה): ").strip()
        if not question:
            break

        # שאלת המשך הופכת לשאלה עצמאית, ושאלה כפולה מתפרקת - ולכל שאלה חיפוש משלה
        queries = build_search_queries(question, history, rewrite_llm)
        documents = retrieve_for_queries(queries, index)

        if SHOW_DOCUMENTS:
            print(f"  חיפוש: {' | '.join(queries)}")
            for doc in documents:
                print(f"  [{doc['score']:.3f}] {doc['source']}")

        answer, sources = generate_answer(question, documents, llms, history, queries)

        print(f"\nעוזר: {answer}")
        if sources:
            print(f"מקורות: {', '.join(sources)}")

        # שומרים את השאלה והתשובה, כדי ששאלת ההמשך תובן בהקשר
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
