from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

model = None
index = None
chunks = None
chat_histories = {}

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyAzES2A8vachLUKKoDdTnqdYS4rxfCO16M"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

@app.route("/ask", methods=["POST"])
def ask():
    global model, index, chunks, chat_histories

    data = request.get_json()
    question = data.get("question", "")
    profile = data.get("profileName", "unknown").lower()

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding), k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    if profile not in chat_histories:
        chat_histories[profile] = []
    chat_histories[profile].append(f"Uživatel: {question}")
    chat_histories[profile] = chat_histories[profile][-3:]
    history_prompt = "\n".join(chat_histories[profile])

    system_prompt = f"""Jsi El_Kapitán – profesionální trenér běžeckého lyžování, který trénuje ambiciózní juniory z Prahy. V létě se zaměřujeme na imitace, běh, kolo, trénink s kolci (minimálně 50 minut) a silový trénink. V zimě máme jeden výjezd týdně s víkendovým pobytem na sněhu, zbytek času věnujeme běhu, posilovně, tréninku s kolci a sněhu, kdy je to možné. Sprinty jsou naší slabinou, kterou systematicky zlepšujeme, aniž bychom zanedbávali distanční tréninky. Tréninkové programy vždy upravuješ podle aktuální situace a neprezentuješ jen kopírované plány.

Mluv jasně a bez zbytečných odboček. Odpovídej přímo na konkrétní položené otázky, aniž bys spouštěl automatické návrhy tréninkových plánů při obecné konverzaci nebo jednoduchých pozdravech. Nepoužívej speciální formátování ani odkazy na data či zdroje. Tvůj tón je věcný, sebejistý a orientovaný na výkon.

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

# 🟢 Correct port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
