
from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ========== FAKE DATA (pro demo - nahradíš svými chunky a embeddingy) ==========
chunks = [
    "PO1 is focused on developing general endurance.",
    "U16 athletes should prioritize ski imitation and terrain running.",
    "Recovery is essential after high-intensity sessions.",
    "Strength training in PO1 focuses on core and coordination.",
    "Training volume should increase gradually over PO1."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ========== Flask + Gemini setup ==========
app = Flask(__name__)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-pro-latest")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    # FAISS retrieval
    query_embedding = model.encode([question])
    _, indices = index.search(np.array(query_embedding), 3)
    context_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(context_chunks)

    # Prompt
    prompt = f"""You are a professional cross-country skiing coach. Use the context below to answer this question practically and clearly.

Context:
{context}

Question: {question}
"""

    response = gemini.generate_content(prompt)
    return jsonify({{"answer": response.text}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
