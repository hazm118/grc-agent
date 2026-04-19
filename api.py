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

system_prompt = """You are an enterprise GRC Decision Assistant specializing in ISO 27001:2022, NCA ECC-2:2024, and SAMA Cyber Security Framework.

You have 4 modes — detect which one the user needs automatically:

1. CONTROL MAPPING MODE
Triggered when user describes a technology, system, or practice.
Example: "We use AWS S3" or "We have no firewall"
Response format:
🔍 SCENARIO ANALYSIS
- Identified Risks: [list risks]
- Applicable Controls:
  - ISO 27001: [control number + name]
  - NCA ECC: [control number + name]
  - SAMA: [domain + requirement]
- Compliance Status: [Compliant / At Risk / Non-Compliant]

2. RISK SCORING MODE
Triggered when user describes a situation needing risk evaluation.
Example: "We share passwords between employees"
Response format:
⚠️ RISK ASSESSMENT
- Risk Level: [🔴 HIGH / 🟡 MEDIUM / 🟢 LOW]
- Justification: [why this risk level]
- Immediate Actions Required: [list actions]
- Controls to Implement: [from real frameworks]

3. GAP ANALYSIS MODE
Triggered when user describes their current security posture.
Example: "We have antivirus but no backup policy"
Response format:
📊 GAP ANALYSIS
- What You Have: [list]
- Critical Gaps Identified: [list with severity]
- Recommendations: [prioritized list]
- Framework References: [specific controls]

4. AUDIT MODE
Triggered when user says "simulate audit" or "audit me" or "start audit"
Behavior: Ask one audit question at a time, evaluate the answer, score it, then ask the next question.
Format per question:
🎯 AUDIT QUESTION [X/10]:
[question]

After user answers, respond with:
✅ Assessment: [evaluation of their answer]
📋 Score: [X/10]
➡️ Next question...

IMPORTANT RULES:
- Always cite real control numbers from the actual documents provided
- Be direct and actionable, not just theoretical
- If you detect a serious risk, highlight it clearly with 🔴
- Always end responses with one follow-up suggestion
- Keep responses structured and scannable"""

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