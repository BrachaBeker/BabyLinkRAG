import os

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

# טוען את המשתנים מקובץ .env אל משתני הסביבה (לפני שקוראים אותם)
load_dotenv()

# הגדרות
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")


def main():
    """בדיקה שהמפתח עובד: שולחים "שלום" למודל ומדפיסים את התשובה."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    llm = GoogleGenAI(model=CHAT_MODEL, api_key=api_key)

    response = llm.complete("שלום")
    print(f"Model: {CHAT_MODEL}")
    print(f"Answer: {response.text}")


if __name__ == "__main__":
    main()
