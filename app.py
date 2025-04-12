from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google.generativeai import configure, GenerativeModel
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# 🌍 Init model + embeddings
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = GenerativeModel("models/gemini-1.5-flash-latest")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 📄 Load chunks & FAISS
with open("chunks.json", "r") as f:
    chunks = json.load(f)

index = faiss.read_index("faiss.index")
embeddings = embedding_model.encode(chunks)

# 👤 Load profiles
with open("profiles.json", "r") as f:
    profiles = json.load(f)

# 💬 Conversation memory per user
memory = {}

# 🧠 Custom system prompt
base_prompt = """
You are El_Kapitán_100b, a professional cross-country skiing coach and the first AI model created by ZDRP_AI. You specialize in training junior athletes aged 18 and above and combine scientific knowledge with practical coaching experience at the elite level.

Be clear, specific, and practical in your responses. Avoid vague phrases like "it depends on the individual." Always provide actionable guidance.

If the user asks a general question like "How should I train in summer?", respond with strategic principles and training concepts—not specific training plans right away.

If the user asks for a specific training session (e.g. sprint, strength, or intervals), provide a detailed training unit example, including warm-up, main set, and cooldown.

Do not refer to where your knowledge comes from or mention any external sources. Act as if the knowledge is your own.
"""

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    user_id = data.get("userId")
    profile_name = data.get("profileName", "unknown").lower()

    if not question or not user_id:
        return jsonify({"error": "Missing question or userId"}), 400

    # 🧠 Personalized context
    profile_context = profiles.get(profile_name, "")
    system_prompt = base_prompt + f"\n\nHere is information about the user: {profile_context.strip()}"

    # 💬 Memory
    history = memory.get(user_id, [])
    history.append({"role": "user", "parts": [question]})
    memory[user_id] = history[-10:]

    # 🔍 RAG
    question_embedding = embedding_model.encode([question])
    D, I = index.search(np.array(question_embedding), k=8)
    retrieved_chunks = "\n\n".join([chunks[i] for i in I[0]])

    prompt = [
        {"role": "system", "parts": [system_prompt]},
        *memory[user_id],
        {"role": "user", "parts": [f"{question}\n\nRelevant info:\n{retrieved_chunks}"]}
    ]

    response = model.generate_content(prompt)
    answer = response.text

    memory[user_id].append({"role": "model", "parts": [answer]})

    return jsonify({"answer": answer})
