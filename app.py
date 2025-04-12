
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vertexai.preview.generative_models import GenerativeModel, Part

app = Flask(__name__)
CORS(app)

# Lazy inicializace
model = None
index = None
chunks = None

# Načtení zkrácených profilů (bodová struktura)
with open("profiles.json", "r") as f:
    profiles = json.load(f)

@app.route("/ask", methods=["POST"])
def ask():
    global model, index, chunks

    data = request.get_json()
    question = data.get("question", "")
    user_id = data.get("userId", "default")
    profile_name = data.get("profileName", "unknown").lower()

    if not question or not user_id:
        return jsonify({"answer": "Chybí otázka nebo ID uživatele."}), 400

    if model is None:
        model = SentenceTransformer("intfloat/e5-small")

    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    # Embed a najdi chunky
    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding), k=3)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    # Personalizace jako body
    profile_bullets = profiles.get(profile_name, [])
    profile_text = "\n".join(profile_bullets)

    # Sestavení promtu
    system_prompt = f"""You are El_Kapitán_100b first AI model from ZDRP AI, a professional cross-country skiing coach. 
You focus on junior athletes and answer based only on the context provided. Do not use special formating

User profile:
{profile_text}

Context:
{context}
"""

    try:
        model_gemini = GenerativeModel("gemini-1.5-flash-preview-0514")
        chat = model_gemini.start_chat()
        response = chat.send_message([
            Part.from_text(system_prompt),
            Part.from_text(f"User: {question}")
        ])
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
