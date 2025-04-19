from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai

# ─── Flask & CORS setup ─────────────────────────────────────
app = Flask(__name__)

# Allow ONLY your Netlify frontend to call /ask
CORS(
    app,
    resources={r"/ask": {"origins": ["https://cosmic-crostata-1c51df.netlify.app"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# Fallback: ensure every response has the right headers
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://cosmic-crostata-1c51df.netlify.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ─── Globals for embeddings & histories ─────────────────────
tokenizer = None
model = None
index = None
chunks = None
chat_histories = {}

# ─── Gemini API setup ───────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ─── Embedding function ────────────────────────────────────
def get_embedding(text):
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained("Seznam/retromae-small-cs")
        model = AutoModel.from_pretrained("Seznam/retromae-small-cs")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0].numpy()

# ─── /ask endpoint ─────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    global index, chunks, chat_histories

    data = request.get_json()
    question = data.get("question", "")
    profile = data.get("profileName", "unknown").lower()

    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    # retrieve top‑5 chunks
    query_embedding = get_embedding(question)
    D, I = index.search(np.array(query_embedding), k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    # manage chat history
    chat_histories.setdefault(profile, []).append(f"Uživatel: {question}")
    chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    # build system prompt
    system_prompt = f"""Jsi El_Kapitán – profesionální trenér běžeckého lyžování...
{context}

Poslední zprávy:
{history_prompt}
"""

    try:
        response = gemini_model.generate_content(system_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {e}"})

# ─── Launch on Render ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
