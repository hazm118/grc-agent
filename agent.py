from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

print("Loading GRC Agent...")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("grc_docs")
print("Ready ✅\n")

conversation_history = []

system_prompt = """You are an expert GRC assistant specializing in ISO 27001, NCA ECC, and SAMA frameworks.
You will be given relevant excerpts from the actual documents to help answer questions.
Always reference specific control numbers when relevant.
Explain in simple, clear words. Be precise and accurate."""

def search_docs(query, n_results=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    chunks = results['documents'][0]
    sources = [m['source'] for m in results['metadatas'][0]]
    context = ""
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        context += f"\n[Source: {source}]\n{chunk}\n"
    return context

print("GRC Agent ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    print("Searching documents...")
    context = search_docs(user_input)

    conversation_history.append({
        "role": "user",
        "content": f"""Answer this GRC question using the document excerpts below.

Question: {user_input}

Relevant document excerpts:
{context}"""
    })

    response = client.chat.completions.create(
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

    # Keep only last 4 messages to stay within free tier limits
    if len(conversation_history) > 4:
        conversation_history = conversation_history[-4:]

    print(f"\nAgent: {reply}\n")