from rag import get_clients, retrieve_documents, generate_answer, NO_ANSWER, TOP_K

# שאלות בדיקה: (שאלה, הקבצים שבהם נמצאת התשובה).
# None = אין על זה מידע במסמכים, והעוזר צריך לענות NO_ANSWER
QUESTIONS = [
    ("איך יוצאים משיבוץ?", ["03_babysitters_guide.md", "04_rules_and_policies.md"]),
    ("איך מפרסמים בקשה לשמרטפית?", ["02_parents_guide.md"]),
    ("מה קורה אם ההורה ביטל בקשה שתפסתי?", ["03_babysitters_guide.md"]),
    ("האם צריך לשלם על השימוש באתר?", ["05_faq.md", "04_rules_and_policies.md"]),
    ("שכחתי את הסיסמה, מה עושים?", ["05_faq.md", "06_contact_and_support.md"]),
    ("מי יכול לראות את מספר הטלפון שלי?", ["04_rules_and_policies.md", "03_babysitters_guide.md"]),
    ("מה התעריף הכי נמוך שאפשר לבקש לשעה?", ["04_rules_and_policies.md", "02_parents_guide.md"]),
    ("איך יוצרים קשר עם התמיכה?", ["06_contact_and_support.md"]),
    ("מה זה BabyLink?", ["01_about_babylink.md"]),
    ("אפשר לבחור שמרטפית מסוימת?", ["05_faq.md"]),
    ("כמה עולה חניה ליד הבית?", None),
]


def main():
    """לכל שאלה: האם הקובץ הנכון נשלף (Recall@k). ובסוף - ציון כולל."""
    llm, index, collection = get_clients()

    found = 0
    answerable = 0

    for question, expected in QUESTIONS:
        documents = retrieve_documents(question, index)
        sources = [doc["source"] for doc in documents]

        # שאלה בלי תשובה במסמכים: בודקים שהמודל עונה NO_ANSWER ולא ממציא
        if expected is None:
            answer, _ = generate_answer(question, documents, llm, [])
            ok = answer == NO_ANSWER
            print(f"{'V' if ok else 'X'} {question}")
            print(f"    expected NO_ANSWER, got: {answer[:80]}")
            continue

        answerable += 1
        hit = any(source in expected for source in sources)
        if hit:
            found += 1

        print(f"{'V' if hit else 'X'} {question}")
        print(f"    expected: {', '.join(expected)}")
        print(f"    retrieved: {', '.join(sources)}")

    print(f"\nRecall@{TOP_K}: {found}/{answerable} = {found / answerable:.0%}")


if __name__ == "__main__":
    main()
