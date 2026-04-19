from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
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

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = []

system_prompt = """You are an enterprise GRC Decision Assistant specializing in ISO 27001:2022, NCA ECC-2:2024, and SAMA Cyber Security Framework.

You have 4 modes — detect which one the user needs automatically:

1. CONTROL MAPPING MODE
Triggered when user describes a technology, system, or practice.
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
Response format:
⚠️ RISK ASSESSMENT
- Risk Level: [🔴 HIGH / 🟡 MEDIUM / 🟢 LOW]
- Justification: [why this risk level]
- Immediate Actions Required: [list actions]
- Controls to Implement: [from real frameworks]

3. GAP ANALYSIS MODE
Triggered when user describes their current security posture.
Response format:
📊 GAP ANALYSIS
- What You Have: [list]
- Critical Gaps Identified: [list with severity]
- Recommendations: [prioritized list]
- Framework References: [specific controls]

4. AUDIT MODE
Triggered when user says simulate audit or audit me or start audit.
Ask one audit question at a time, evaluate the answer, score it, then ask the next.
Format:
🎯 AUDIT QUESTION [X/10]: [question]
After answer:
✅ Assessment: [evaluation]
📋 Score: [X/10]

IMPORTANT RULES:
- Always cite real ISO 27001:2022, NCA ECC-2:2024, and SAMA control numbers
- Be direct and actionable
- Highlight serious risks with 🔴
- End responses with one follow-up suggestion
- Keep responses structured and scannable"""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "GRC Agent API is running ✅"}

@app.post("/chat")
def chat(request: ChatRequest):
    global conversation_history

    conversation_history.append({
        "role": "user",
        "content": request.message
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