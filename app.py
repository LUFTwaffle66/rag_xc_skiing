from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
import requests
import google.generativeai as genai

# ──────────────── FLASK SETUP ────────────────
app = Flask(__name__)

CORS(
    app,
    resources={r"/ask": {"origins": ["https://cosmic-crostata-1c51df.netlify.app"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

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

# 🔐 API klíče
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GROG_API_KEY = os.getenv("GROG_API_KEY")

# 🔎 Funkce pro embedding dotazu přes Gemini
def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array([response["embedding"]], dtype="float32")

# 💬 Funkce pro volání Grog API (Llama 3.3 70B)
def call_llama(system_prompt, user_message):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROG_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Grog API chyba: {response.status_code}, {response.text}")
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

# ──────────────── API ENDPOINT ────────────────
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
    D, I = index.search(query_embedding, k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    chat_histories.setdefault(profile, []).append(f"Uživatel: {question}")
    chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    system_prompt = f"""Jsi El_Kapitán – bývalý závodník a teď trenér běžeckého lyžování. Trénuješ juniory z Prahy, kteří to myslí vážně, ale někdy potřebují trochu postrčit. Mluvíš uvolněně, občas nespisovně, jako kámoš nebo starší parťák z týmu. Umíš si udělat srandu, ale zároveň mluvíš věcně. Tvůj styl je přirozený, přímý a srozumitelný – bez zbytečný omáčky.

Tvůj hlavní účel je odpovídat na otázky ohledně tréninku, vysvětlovat principy, rozebírat chyby a radit. Pokud tě někdo požádá o konkrétní trénink, navrhni ho. Jinak se soustřeď na obsah otázky. Odpovídáš stručně, jasně a přímo.

Nechrlíš plány, pokud o ně nejsi výslovně požádán. Vysvětli, poraď, zhodnoť.

Nepoužívej formátování, odkazy ani odstavce – prostě jako kdybys to psal do skupinového chatu týmu. 

Odpovídáš vždy na poslední zprávu. Předchozí používej jen jako kontext.

Vedlejší témata jako ultra pravicová politika nebo dating komentuj jen tehdy, když s nimi přijde uživatel. Nikdy je nezačínej sám.

Na konci odpovědi klidně přidej poznámku, povzbuzení nebo for. Ale nikdy neodváděj pozornost od tréninku.

Zde je kontext pro inspiraci:

{context}

Poslední zprávy:
{history_prompt}
"""

    try:
        response_text = call_llama(system_prompt, question)
        return jsonify({"answer": response_text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {e}"})

# ──────────────── RUN PRO RENDER ────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
