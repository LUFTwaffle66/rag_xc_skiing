from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
import google.generativeai as genai

# ──────────────── FLASK SETUP ────────────────
app = Flask(__name__)

# ✅ Povolit volání jen z tvé Netlify stránky
CORS(
    app,
    resources={r"/ask": {"origins": ["https://cosmic-crostata-1c51df.netlify.app"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# (Nepovinná záloha CORS hlaviček)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://cosmic-crostata-1c51df.netlify.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ──────────────── PROMĚNNÉ ────────────────
index = None
chunks = None
chat_histories = {}

# 🔐 Gemini API klíč
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# 🔎 Funkce pro embedding dotazu přes Gemini
def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array([response["embedding"]], dtype="float32")

# ──────────────── API ENDPOINT ────────────────
@app.route("/ask", methods=["POST"])
def ask():
    global index, chunks, chat_histories

    data = request.get_json()
    question = data.get("question", "")
    profile = data.get("profileName", "unknown").lower()

    # 🧠 Načíst index a texty, pokud ještě nejsou načtené
    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    # 🔍 Vyhledat relevantní části
    query_embedding = get_embedding(question)
    D, I = index.search(query_embedding, k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    # 💬 Správa historie dotazů
    chat_histories.setdefault(profile, []).append(f"Uživatel: {question}")
    chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    # 🧠 Vytvoření proměnné pro celý prompt
    system_prompt = f"""Jsi El_Kapitán – profesionální trenér běžeckého lyžování. Trénuješ ambiciózní juniory z Prahy, kteří to myslí vážně. Reaguj stručně, bez výmluv a bez omáčky. Nepoužívej fráze jako 'záleží', rozhodni se sám.

Zde je relevantní kontext, nemusíš vycházet pouze z toho, spíš se inspiruj:
{context}

Poslední zprávy:
{history_prompt}
"""

    try:
        response = gemini_model.generate_content(system_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {e}"})

# ──────────────── RUN PRO RENDER ────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
