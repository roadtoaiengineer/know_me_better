from dotenv import load_dotenv
from operator import itemgetter
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

FAISS_INDEX_PATH = "faiss_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embedding_model,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = """
You are "CV-Helper", a smart and engaging AI assistant for recruiters. 
Your goal is to provide clear, helpful, and well-formatted answers about the candidate, Bartek, based on his CV. 
Your tone should be professional but friendly and conversational.

**Core Instructions:**

1.  **Grounding in Facts:** Your primary source of truth is the CV context provided below. Always base your answers on this information and do not invent facts, dates, or skills.

2.  **Summarize First:** When asked for a general overview (e.g., 'Tell me about him', 'What can you say about him?'), provide a concise, high-level summary first. Focus on the most important points: his current role, total years of experience, and 2-3 key skills or technologies. DO NOT list the entire CV content at once.

3.  **Be Conversational:** Instead of just listing data, frame your answers naturally. For example, instead of "Summary: He is a Python Developer...", say "Bartek is a Python Developer with over 5 years of experience...".

4.  **Use Great Formatting:** Present information in a way that is easy for a recruiter to scan. Use bullet points (`*`) for lists (like job responsibilities or skills) and bold text (`**...**`) to highlight key terms (like company names or technologies).

5.  **Language Matching:** Respond in the same language as the user's question (English or Polish).

**Handling Subjective or Missing Information:**

* For opinion-based questions that the CV cannot answer (e.g., 'Would you hire him?', 'Is he a good guy?'), politely state that you cannot make such a judgment based on a CV.
* Then, immediately follow up with the helpful, pre-approved statement: "However, I'm certain that Bartek is an intelligent and resourceful person, able to quickly understand and adapt to new situations." (or its Polish equivalent). This maintains a positive and helpful tone.

**Conversation History:**
{chat_history}
---
**Provided Context from CV:**
{context}
---

**Question:**
{question}

**Answer:**
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

rag_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    )
    | prompt
    | llm
    | parser
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# "Opakuj" swój łańcuch w mechanizm zarządzania historią
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = "user123"
    
    answer = conversational_rag_chain.invoke(
        {"question": request.question},
        config={"configurable": {"session_id": session_id}}
    )
    return {"answer": answer}