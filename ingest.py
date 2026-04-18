from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

try:
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded ✅")

    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("grc_docs")
    except:
        pass

    collection = client.get_or_create_collection("grc_docs")
    print("Database ready ✅")

    docs_folder = "./docs"
    pdf_files = {
        "iso27001 2022.pdf": "ISO27001",
        "ECC--2024-EN.pdf": "NCA_ECC",
        "SAMA Cyber Security Framework.pdf": "SAMA"
    }

    for filename, source_name in pdf_files.items():
        filepath = os.path.join(docs_folder, filename)
        print(f"Reading {source_name} from {filepath}...")

        if not os.path.exists(filepath):
            print(f"  ❌ FILE NOT FOUND: {filepath}")
            continue

        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        print(f"  Extracted {len(full_text)} characters")

        chunks = []
        chunk_size = 500
        words = full_text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

        print(f"  Creating embeddings for {len(chunks)} chunks...")
        embeddings = model.encode(chunks).tolist()

        ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name} for _ in chunks]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"  ✅ {source_name} done — {len(chunks)} chunks stored")

    print("\n✅ All documents loaded into database!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()