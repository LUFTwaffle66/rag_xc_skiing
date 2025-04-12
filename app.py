from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from vertexai.preview.generative_models import GenerativeModel, Part
import logging # Added for better logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Lazy Initialization Globals ---
embedding_model = None
faiss_index = None
text_chunks = None
generative_model = None
# --- --- --- --- --- --- --- --- ---

# Load profiles (assuming this is small and okay at startup)
try:
    with open("profiles.json", "r") as f:
        profiles = json.load(f)
    logger.info("Profiles loaded successfully.")
except FileNotFoundError:
    logger.error("profiles.json not found! Personalization will be limited.")
    profiles = {}
except json.JSONDecodeError:
    logger.error("Error decoding profiles.json! Personalization will be limited.")
    profiles = {}


# --- Helper function for lazy loading ---
def initialize_models():
    global embedding_model, faiss_index, text_chunks, generative_model

    if embedding_model is None:
        logger.info("Initializing Sentence Transformer model...")
        try:
            # You might consider an even smaller model if needed, e.g., 'all-MiniLM-L6-v2'
            # but e5-small is a good balance.
            embedding_model = SentenceTransformer("intfloat/e5-small")
            logger.info("Sentence Transformer model initialized.")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}", exc_info=True)
            # Decide how to handle this - maybe raise the exception or exit?
            # For now, we'll let requests fail later if the model is needed.

    if faiss_index is None:
        logger.info("Initializing FAISS index...")
        try:
            # *** Optimization 1: Use Memory Mapping ***
            faiss_index = faiss.read_index("faiss.index", faiss.IO_FLAG_MMAP)
            logger.info("FAISS index initialized with memory mapping.")
            # *** Optimization 2 (Potential): Load chunks only if index loads ***
            # If chunks.json is large, consider alternative loading here.
            # For now, loading it fully after index load.
            logger.info("Loading text chunks...")
            with open("chunks.json", "r") as f:
                text_chunks = json.load(f)
            logger.info("Text chunks loaded.")
        except FileNotFoundError as e:
             logger.error(f"Failed to load FAISS index or chunks: {e}. Check faiss.index and chunks.json exist.")
        except Exception as e:
            logger.error(f"Error initializing FAISS index or loading chunks: {e}", exc_info=True)


    if generative_model is None:
        logger.info("Initializing Vertex AI Generative Model...")
        try:
            # *** Optimization 3: Reuse GenerativeModel instance ***
            generative_model = GenerativeModel("gemini-1.5-flash-preview-0514")
            logger.info("Vertex AI Generative Model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI model: {e}", exc_info=True)

# --- --- --- --- --- --- --- --- ---

@app.route("/ask", methods=["POST"])
def ask():
    # --- Ensure models are loaded ---
    try:
        initialize_models()
    except Exception as e:
        # If initialization itself failed critically
        logger.error(f"Critical initialization error: {e}", exc_info=True)
        return jsonify({"answer": "Server error during initialization."}), 500

    # --- Check if essential components loaded ---
    if embedding_model is None or faiss_index is None or text_chunks is None or generative_model is None:
         logger.error("One or more essential models/data failed to load. Cannot process request.")
         return jsonify({"answer": "Server configuration error: Models not loaded."}), 503 # Service Unavailable

    # --- Process Request ---
    data = request.get_json()
    if not data:
        return jsonify({"answer": "Invalid request body."}), 400

    question = data.get("question", "")
    user_id = data.get("userId", "default") # Consider actually using userId if needed later
    profile_name = data.get("profileName", "unknown").lower()

    if not question:
        return jsonify({"answer": "Chybí otázka."}), 400
    # Removed userId check as it wasn't used beyond logging/potential future use

    logger.info(f"Received question: '{question}' for profile: '{profile_name}'")

    try:
        # 1. Embed question
        query_embedding = embedding_model.encode([question])

        # 2. Search FAISS
        # Using k=3 seems reasonable. Adjust if needed.
        k_neighbors = 3
        distances, indices = faiss_index.search(np.array(query_embedding), k=k_neighbors)

        # 3. Get relevant chunks
        relevant_chunks = []
        if indices.size > 0:
            for i in indices[0]:
                if 0 <= i < len(text_chunks): # Check index validity
                     relevant_chunks.append(text_chunks[i])
                else:
                    logger.warning(f"FAISS returned invalid index: {i}, skipping.")
        else:
             logger.warning("FAISS search returned no indices.")

        context = "\n".join(relevant_chunks)
        logger.debug(f"Retrieved context: {context[:200]}...") # Log snippet

        # 4. Get profile personalization
        profile_bullets = profiles.get(profile_name, [])
        profile_text = "\n".join(profile_bullets)
        logger.debug(f"Using profile text: {profile_text[:200]}...") # Log snippet

        # 5. Build Prompt
        system_prompt = f"""You are El_Kapitán_100b first AI model from ZDRP AI, a professional cross-country skiing coach.
You focus on junior athletes and answer based only on the context provided. Do not use special formatting. Do not copy paste trainings directly, but explain concepts based on them.

User profile:
{profile_text}

Context:
{context}
"""
        user_message = f"User: {question}"

        # 6. Call Generative Model
        # Start chat for potential multi-turn, though only one turn used here
        chat = generative_model.start_chat()
        response = chat.send_message([
            Part.from_text(system_prompt),
            Part.from_text(user_message)
        ])

        logger.info(f"Successfully generated response for question: '{question}'")
        return jsonify({"answer": response.text})

    except Exception as e:
        logger.error(f"Error processing question '{question}': {str(e)}", exc_info=True)
        # Provide a generic error message to the user
        return jsonify({"answer": "Omlouváme se, při zpracování vašeho dotazu došlo k chybě."}), 500

# Note: Remove debug=True when deploying!
# Use a proper WSGI server like gunicorn:
# gunicorn -w 1 --bind 0.0.0.0:$PORT app:app
if __name__ == "__main__":
    # Make sure models load on start when running directly for testing
    initialize_models()
    # Port configuration often comes from environment variables in deployment
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port) # Set debug=False, host needed for container access
