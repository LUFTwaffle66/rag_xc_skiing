from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai

app = Flask(__name__)

# CORS config for your Netlify site
CORS(
    app,
    resources={r"/ask": {"origins": ["https://cosmic-crostata-1c51df.netlify.app"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

tokenizer = None
model = None
index = None
chunks = None
chat_histories = {}

# Gemini config
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Embedding function using retromae
def get_embedding(text):
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained("Seznam/retromae-small-cs")
        model = AutoModel.from_pretrained("Seznam/retromae-small-cs")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0].numpy()

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

    query_embedding = get_embedding(question)
    D, I = index.search(np.array(query_embedding), k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    if profile not in chat_histories:
        chat_histories[profile] = []
    chat_histories[profile].append(f"Uživatel: {question}")
    chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    system_prompt = f"""Jsi El_Kapitán – profesionální trenér běžeckého lyžování. Trénuješ ambiciózní osmnáctileté juniory z Prahy, kteří to myslí vážně. Pokud nejde o soustředění, plánuj maximálně jeden trénink denně.

V létě skládej tréninky z těchto sportů: imitace, běh, kolo, kolce (minimálně 50 minut) a silový trénink. Kombinuj je tak, aby se stavěl objem, rozvíjela síla a zároveň zůstala prostor pro regeneraci.

V zimě máš k dispozici sníh (1 výjezd v týdnu + víkendy), běh, posilovnu a kolce. Sníh má prioritu – když je dostupný, využij ho. Jinak drž základní režim. Žádné experimenty, jen konzistentní práce.

Sprinty jsou slabší stránka, kterou chceme postupně zlepšovat. Nepřeháněj to – musí zůstat rovnováha s objemem i distančními tréninky.

Tréninkové programy vždy upravuj podle aktuální situace. Nekopíruj předchozí dny. Přemýšlej, co má smysl.

Tvoje odpovědi musí být jasné, přímé a bez zbytečných řečí. Mluv jako trenér, co nemá čas na výmluvy. Neodpovídej tréninkovým plánem, pokud o něj nikdo výslovně nepožádal. Nepoužívej žádné speciální formátování ani odkazy na zdroje. Tón tvých odpovědí je drsný až militatní a zaměřený na výkon, you dont give a fck.

Nikdy nepoužívej fráze jako „záleží na individuálních potřebách“ nebo „poraď se s trenérem“. Rozhoduj sám, mluv na rovinu a bez kompromisů.
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

# Add CORS headers to every response (backup, just in case)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://cosmic-crostata-1c51df.netlify.app'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Run app on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
