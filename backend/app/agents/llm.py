from langchain_groq import ChatGroq
from app.config import settings

GroqModel = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=0,
    api_key=settings.GROQ_API_KEY
)