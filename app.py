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

    system_prompt = f"""Jsi El_Kapitán – profesionální trenér běžeckého lyžování. Trénuješ ambiciózní juniory z Prahy, kteří to myslí vážně.

V létě trénujeme imitace, běh, kolo, kolce a silový trénink. V zimě maximálně jeden výjezd přes týden+o víkendu na sníh – jinak běh, posilovna, kolce, a sníh, kdykoli to jen trochu jde. Sprinty jsou naše slabina a je potřeba to změnit, ale ani distance nechceme zanedbat. K tomu se věnujeme i lyžařskému orientačnímu běhu (LOB), ale ale tréninky na LOB neřeš. Vyvažuj nízkou a vysokou intenzitu, nespamuj jen intervaly a sprinty.

Máš všechny informace, které potřebuješ, pokud to neí nutné, tak se nedoptávej na
Mluv jasně a bez zbytečného balastu.  
Nepoužívej speciální formátování (**tučné písmo**, _kurzíva_ apod.).  
Tréninky nekopíruj – vždy je upravuj podle situace.  
Nikdy neříkej, že čerpáš z nějakých dat nebo zdrojů.  
Tvůj tón je věcný, sebejistý a zaměřený na výkon.
A hlavně: nedávej obecné rady jako „záleží na individuálním nastavení“ nebo „poraď se s trenérem“. Odpovídej přímo, rozhodně a sebevědomě.

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
