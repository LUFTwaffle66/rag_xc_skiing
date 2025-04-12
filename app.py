
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- Init Flask ---
app = Flask(__name__)
CORS(app, origins=["https://cosmic-crostata-1c51df.netlify.app"], supports_credentials=True)

# --- Load data ---
with open("chunks.json", "r") as f:
    chunks = json.load(f)

index = faiss.read_index("faiss.index")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Gemini setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-1.5-pro-latest")

# --- RAG endpoint ---
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    # Embed question + search FAISS
    query_embedding = embedder.encode([question])
    _, indices = index.search(np.array(query_embedding), k=5)
    context_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(context_chunks)

    # Prompt
    prompt = f"""
You are a professional cross-country skiing coach named El_Kapitán_100b – the first AI model created by ZDRP_AI.

You specialize in training junior athletes aged 18+, combining scientific knowledge with hands-on coaching experience at the elite level.

Your answers must be:
- Clear, specific, and practical
- Easy to understand
- Free of vague or generic phrases (do not say things like “it depends on individual preferences”)

Only respond based on the context provided. 
Do not include your own assumptions, external knowledge, or general advice — rely solely on the content available in the context.

When appropriate, include a concrete example of a training session or a specific workout.

Always respond in Czech if the question is asked in Czech. Otherwise, reply in English.

---

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = model_gemini.generate_content(prompt)
    return jsonify({"answer": response.text})

# --- Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
