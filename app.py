from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Inicializace globálních proměnných
model = None
index = None
chunks = None
chat_histories = {}

# Nastavení Gemini API
genai.configure(api_key=os.getenv("AIzaSyAzES2A8vachLUKKoDdTnqdYS4rxfCO16M"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-001")

@app.route("/ask", methods=["POST"])
def ask():
    global model, index, chunks, chat_histories

    data = request.get_json()
    question = data.get("question", "")
    profile = data.get("profileName", "unknown").lower()

    # Lazy loading modelu
    if model is None:
        model = SentenceTransformer("intfloat/e5-small")

    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    # Embed dotazu a vyhledání relevantního kontextu
    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding), k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    # Historie konverzace
    if profile not in chat_histories:
        chat_histories[profile] = []
    chat_histories[profile].append(f"Uživatel: {question}")
    if len(chat_histories[profile]) > 3:
        chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    # System prompt
    system_prompt = f"""Jsi El_Kapitán_100b, profesionální trenér běžeckého lyžování.
Nepoužívej speciální formátování. Nekopíruj tréninky, upravuj je podle situace.
Odpovídáš na základě následujícího kontextu:

{context}

Poslední zprávy z konverzace:
{history_prompt}
"""

    # Volání Gemini API
    try:
        chat = gemini_model.start_chat(history=[])
        response = chat.send_message(system_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
