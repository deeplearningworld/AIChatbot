# --- ADDED: The Fix for the macOS Crash ---
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
# --- End of Fix ---

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

app = FastAPI(
    title="Document-Based Text Generation API",
    description="An API to generate text based on a pre-defined document about the solar system.",
    version="1.0.0"
)

class PromptRequest(BaseModel):
    prompt: str

def split_text(text, chunk_size=300, chunk_overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def setup_ai_components(text):
    print("Loading AI components...")
    chunks = split_text(text)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
    retriever = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='euclidean')
    retriever.fit(embeddings)
    
    # --- MODIFIED: Switched back to the instruction-tuned flan-t5-base model ---
    print("Loading generator model (flan-t5-base)...")
    generator_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    
    print("AI components loaded successfully!")
    return chunks, embedding_model, retriever, generator_pipeline

document_text = """
The Solar System: A Brief Overview
Our solar system is a vast and fascinating place, anchored by our Sun, a yellow dwarf star that accounts for 99.86% of the system's mass. Orbiting this star are eight planets, numerous dwarf planets, and countless smaller bodies like asteroids and comets.
The planets are divided into two main groups. The inner, rocky planets are Mercury, Venus, Earth, and Mars. They are smaller and primarily composed of rock and metal. Earth is unique among these for its vast oceans of liquid water and the life it supports. Mars is known for its red color, which comes from iron oxide on its surface, and scientists are actively searching for signs of past or present life there.
Beyond Mars lies the asteroid belt, a region filled with rocky bodies that failed to form a planet. The outer planets are the gas giants, Jupiter and Saturn, and the ice giants, Uranus and Neptune. Jupiter is the largest planet in our solar system, famous for its Great Red Spot, a storm larger than Earth. Saturn is renowned for its spectacular ring system, composed of ice and rock particles. Uranus and Neptune are the coldest planets, composed of rock, ice, and a thick mixture of water, ammonia, and methane.
Beyond Neptune is the Kuiper Belt, a region of icy bodies, which is home to the dwarf planet Pluto. For many years, Pluto was considered the ninth planet, but it was reclassified in 2006 because it did not meet all the criteria to be defined as a planet. The solar system extends far beyond the Kuiper Belt to the Oort Cloud, a theoretical sphere of icy bodies that is thought to be the origin of most long-period comets.
"""

chunks, embedder, retriever, generator = setup_ai_components(document_text)

@app.post("/generate/")
async def generate_text(request: PromptRequest):
    user_prompt = request.prompt
    
    query_embedding = embedder.encode([user_prompt])
    _, indices = retriever.kneighbors(query_embedding)
    context = " ".join([chunks[i] for i in indices[0]])
    
    # --- MODIFIED: Using the structured prompt that works best for T5 models ---
    final_prompt = f"""
    Based on the context below, please provide a clear and concise answer to the question.

    Context:
    "{context}"

    Question:
    "{user_prompt}"

    Answer:
    """
    
    # --- MODIFIED: Added parameters to prevent repetition and improve quality ---
    result = generator(
        final_prompt, 
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2, # Prevents repeating the same sequence of 2 words
        early_stopping=True
    )
    
    answer = result[0]['generated_text']
    
    return {"generated_answer": answer, "retrieved_context": context}

if __name__ == "__main__":
    print("Starting backend server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)