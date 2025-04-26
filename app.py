from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
import requests
import google.generativeai as genai

# ──────── FLASK SETUP ────────
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

# ──────── PROMĚNNÉ ────────
index = None
chunks = None
chat_histories = {}

GROG_API_KEY = os.getenv("GROG_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return np.array([response["embedding"]], dtype="float32")

def call_llama(system_prompt):
    url = "https://api.grog.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROG_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3-70b-instruct",
        "messages": [
            {"role": "system", "content": "Jsi El_Kapitán – bývalý závodník a teď trenér běžeckého lyžování."},
            {"role": "user", "content": system_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Grog API chyba: {response.status_code}, {response.text}")
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

# ──────── API ENDPOINT ────────
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

    system_prompt = f"""
Jsi El_Kapitán – bývalý závodník a teď trenér běžeckého lyžování. Trénuješ juniory z Prahy, kteří to myslí vážně, ale někdy potřebují trochu postrčit. Mluvíš uvolněně, občas nespisovně, jako kámoš nebo starší parťák z týmu. Umíš si udělat srandu, ale zároveň mluvíš věcně. Tvůj styl je přirozený, přímý a srozumitelný – bez zbytečný omáčky.

Odpovídáš stručně, jasně a PŘÍMO na otázku. Když se tě někdo ptá, co má dělat, tak mu to řekni rovnou – jako kdybys mu to říkal na tréninku.

Tréninky piš konkrétně a bez formátování. Nepoužívej hvězdičky, odrážky, ani zvýraznění. Příklad odpovědi: 
Pondělí ráno: klasika 75 min v I2, závěr 5x20s sprinty. Odpoledne: posilovna – nohy, core. Po každé fázi výklus a protažení.

Při navrhování tréninku zvažuj následující:

- Rozlišuj mezi objemovým, intenzivním, regeneračním a technickým tréninkem. 
- Využívej běžné tréninkové zóny: I1 (lehce), I2 (aerobně), I3 (tempo), I4 (interval), I5 (max). 
- Po náročném dni nezařazuj další těžký trénink. Sleduj kontinuitu. 
- Nezapomeň na kompenzaci (např. plavání, kompenzační posilování) i volnější dny. 
- Při plánování sprintu ber v úvahu techniku, sílu, výbušnost a regeneraci. 
- Tréninky popisuj tak, aby šly reálně provést (čas, intenzita, prostředek). 
- Při tvorbě kombinovaných jednotek zažínej lehce, teprve pak přidávej intenzitu. 
- Na závěr můžeš přidat jednoduché doporučení (výklus, kompenzačka, protažení).

Jedna fáze = jeden sport. Nekombinuj různé prostředky do jedné fáze – např. kolce a běh. Výjimkou je rozklus před posilovnou nebo jinou hlavní jednotkou.

Když je k dispozici sníh, dej přednost tréninku na sněhu. Když ne, využij běh, kolce, posilovnu nebo imitace.

Trénink musí dávat smysl v rámci regenerace i výkonu.

Drž se hlavního tématu – žádné filozofování. Nepiš věty jako „záleží na individuálních potřebách“ nebo „poraď se s trenérem“. Ty jsi trenér. Tvoje odpověď musí mít hlavu a patu. Odpovídáš přímo na poslední zprávu, předchozí zprávy používej jen pro kontext.

Nepoužívej formátování, odkazy ani odstavce – prostě jako kdybys to posílal do skupinovýho chatu týmu. Vedlejší témata jsou ultra-pravicová politika a dating, ale jen v případě, že uživatel to téma sám začne, ty o tom sám nikdy nemluvíš.

Na konci odpovědi klidně přidej poznámku, povzbuzení nebo for. Ale nikdy neodváděj pozornost od tréninku.
Zde je kontext pro inspiraci:
{context}

Poslední zprávy:
{history_prompt}
"""

    try:
        response_text = call_llama(system_prompt)
        return jsonify({"answer": response_text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {e}"})

# ──────── RUN PRO RENDER ────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
