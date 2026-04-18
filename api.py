from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading GRC Agent...")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("grc_docs")
print("Ready ✅")

conversation_history = []

system_prompt = """You are an expert GRC assistant specializing in ISO 27001, NCA ECC, and SAMA frameworks.
You will be given relevant excerpts from the actual documents to help answer questions.
Always reference specific control numbers when relevant.
Explain in simple, clear words. Be precise and accurate."""

class ChatRequest(BaseModel):
    message: str

def search_docs(query, n_results=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    chunks = results['documents'][0]
    sources = [m['source'] for m in results['metadatas'][0]]
    context = ""
    for chunk, source in zip(chunks, sources):
        context += f"\n[Source: {source}]\n{chunk}\n"
    return context

@app.get("/")
def root():
    return {"status": "GRC Agent API is running ✅"}

@app.post("/chat")
def chat(request: ChatRequest):
    global conversation_history

    context = search_docs(request.message)

    conversation_history.append({
        "role": "user",
        "content": f"""Answer this GRC question using the document excerpts below.

Question: {request.message}

Relevant document excerpts:
{context}"""
    })

    if len(conversation_history) > 4:
        conversation_history = conversation_history[-4:]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt}
        ] + conversation_history,
        max_tokens=1024
    )

    reply = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return {"reply": reply}