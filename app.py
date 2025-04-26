from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai

# ──────── FLASK SETUP ────────
app = Flask(__name__)

# ✅ Povolit volání jen z tvé Netlify stránky
CORS(
    app,
    resources={r"/*": {"origins": ["https://cosmic-crostata-1c51df.netlify.app"]}},
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

# ✅ Obsluha root URL pro preflight requesty
@app.route("/", methods=["GET", "OPTIONS"])
def home():
    return "OK", 200

# ──────── PROMĚNNÉ ────────
index = None
chunks = None

# 🔐 Načtení modelu Seznam/retromae-small-cs pro embedding
embedding_tokenizer = AutoTokenizer.from_pretrained("Seznam/retromae-small-cs")
embedding_model = AutoModel.from_pretrained("Seznam/retromae-small-cs")

# 🔐 Gemini API klíč
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# 🔎 Funkce pro embedding dotazu přes Seznam model
def get_embedding(text):
    with torch.no_grad():
        inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # vezmeme CLS token
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

# (pokračuje endpoint /ask, atd.)
