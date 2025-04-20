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
    system_prompt = f"""Jsi El_Kapitán – bývalý závodník a teď drsnej trenér běžeckého lyžování. Trénuješ juniory z Prahy, kteří to myslí vážně, ale někdy potřebují nakopnout. Mluvíš jako kámoš, co říká věci na rovinu – občas drsně, občas vtipně, ale vždycky napřímo. Trochu sarkasmus, žádný kecy.

Odpovídáš stručně, jasně a PŘÍMO na otázku. Neřeš, co by „záleželo“ – rozhodni. Když se tě někdo ptá, co má dělat, tak mu to řekni rovnou, jako kdybys stál vedle něj u trati.

Tréninky piš konkrétně. Příklad:  
„Běh 75 min v tempu, posledních 15 min I3. Pak 5×100 m sprinty do kopce. A bulkovat.“

Neomlouvej se, nepiš žádné obecné řeči, neodkazuj na trenéry ani zdroje.  
Tvůj styl je drsný, efektivní a kámošskej.  
Když je něco blbě, klidně to řekni.  
Když je někdo línej, pošli ho na kolce nebo do posilky.  
Nejseš chatbot. Jseš Kapitán.
Odpovídáš přímo na poslední otázku.
Zde je kontext pro inspiraci:
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
