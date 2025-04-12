from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Flask app setup
app = Flask(__name__)
CORS(app)

# Lazy-loaded resources
model = None
index = None
chunks = None
chat_histories = {}

# Gemini setup
genai.configure(api_key=os.getenv("AIzaSyAzES2A8vachLUKKoDdTnqdYS4rxfCO16M"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

@app.route("/ask", methods=["POST"])
def ask():
    global model, index, chunks, chat_histories

    data = request.get_json()
    question = data.get("question", "")
    profile = data.get("profileName", "unknown").lower()

    if model is None:
        model = SentenceTransformer("intfloat/e5-small")
    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    # Search relevant chunks
    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding), k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    # Manage conversation history
    if profile not in chat_histories:
        chat_histories[profile] = []
    chat_histories[profile].append(f"Uživatel: {question}")
    chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    # Construct prompt
    system_prompt = f"""Jsi El_Kapitán_100b, profesionální trenér běžeckého lyžování. 
Nepoužívej speciální formátování. Nekopíruj tréninky, upravuj je podle situace. 
Odpovídáš na základě následujícího kontextu:

{context}

Poslední zprávy z konverzace:
{history_prompt}
"""

    try:
        response = gemini_model.generate_content(system_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
