
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

app = Flask(__name__)
CORS(app)

model = None
index = None
chunks = None

@app.route("/ask", methods=["POST"])
def ask():
    global model, index, chunks

    data = request.get_json()
    question = data.get("question", "")

    if model is None:
        model = SentenceTransformer("intfloat/e5-small")

    if index is None:
        index = faiss.read_index("faiss.index")
        with open("chunks.json", "r") as f:
            chunks = json.load(f)

    query_embedding = model.encode([question])
    D, I = index.search(np.array(query_embedding), k=5)
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(relevant_chunks)

    system_prompt = f"""Jsi El_Kapitán_100b, profesionální trenér běžeckého lyžování. Odpovídáš na základě následujícího kontextu:

{context}
"""

    from vertexai.preview.generative_models import GenerativeModel, Part
    model_gemini = GenerativeModel("gemini-1.5-flash-preview-0514")
    chat = model_gemini.start_chat()

    try:
        response = chat.send_message([
            Part.from_text(system_prompt),
            Part.from_text(f"Uživatel: {question}")
        ])
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"answer": f"Chyba: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
