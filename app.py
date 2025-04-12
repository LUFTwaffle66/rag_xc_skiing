
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

# --- Load user profiles ---
with open("profiles.json", "r") as f:
    user_profiles = json.load(f)

# --- Gemini setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-1.5-pro-latest")

# --- RAG endpoint with user profile awareness ---
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    messages = data.get("messages", [])
    user_id = data.get("userId", "unknown").lower()

    # Get last user message
    question = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            question = msg["content"]
            break

    if not question:
        return jsonify({"error": "No user message found."}), 400

    # FAISS retrieval
    query_embedding = embedder.encode([question])
    _, indices = index.search(np.array(query_embedding), k=5)
    context_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(context_chunks)

    # Conversation history
    dialogue = ""
    for msg in messages:
        role = "You" if msg["role"] == "user" else "El_Kapitán_100b"
        dialogue += f"{role}: {msg['content']}\n"

    # User profile
    user_profile = user_profiles.get(user_id, "This user has no specific profile information.")

    # Prompt
    prompt = f"""
You are a professional cross-country skiing coach named El_Kapitán_100b – the first AI model created by ZDRP_AI.

You specialize in coaching junior athletes aged 18+, combining scientific training methodology with practical, field-tested experience.

Your coaching style is:
- Clear, supportive, and realistic
- Focused on endurance, sprint training, strength, recovery, and periodization
- Personal, as if speaking directly to the athlete you're helping

You always:
- Answer ONLY based on the CONTEXT provided – do not invent facts or use external data
- Never copy-paste training plans – you may summarize or rephrase them for clarity
- Use accurate terminology (e.g., PO1, sprint, aerobic capacity, SR, STP, LY) where relevant
- Avoid vague statements like “it depends”

Adjust your response to the user’s question:
- For general questions (e.g., “How should I train in summer?”), provide a structured overview and key concepts
- For specific questions (e.g., “Give me a sprint session”), give a clearly detailed training example
- Do not use Markdown-style formatting – answer with clean, natural paragraphs

Always respond in Czech if the user speaks Czech. Otherwise, answer in English.

---

The current user is: {user_id}
Profile:
{user_profile}

---

CONTEXT:
{context}

CONVERSATION:
{dialogue}

ANSWER THE LAST USER QUESTION ABOVE:
"""

    response = model_gemini.generate_content(prompt)
    return jsonify({"answer": response.text})

# --- Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
