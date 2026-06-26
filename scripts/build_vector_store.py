import os
import glob
import json
import faiss
import pickle
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


# Configuration & Paths

# CHUNK_DIR: Absolute path to the directory containing the chunked JSON files.
CHUNK_DIR = r"C:\Users\dauda\OneDrive\Desktop\LLAMA_4_LEGAL_V2_REAL\training_data\chunks"

# OUTPUT_DIR: Absolute path to the directory where the FAISS index and metadata will be saved.
OUTPUT_DIR = r"C:\Users\dauda\OneDrive\Desktop\LLAMA_4_LEGAL_V2_REAL\vector_store"

# Ensure the output directory exists.
os.makedirs(OUTPUT_DIR, exist_ok=True)     


# 1. Load Data Chunks

print("loading chunks...")

# glob.glob finds all files matching the given pattern (all .json files in the chunk directory).
chunk_files = glob.glob(os.path.join(CHUNK_DIR, "*.json"))   
print("found:", len(chunk_files))

# Initialize an empty list to store the chunk data.
# Each item will be a dictionary containing the chunk's text and its source file.
chunks = []    


# Loop through each located JSON chunk file.
for p in chunk_files:
    with open(p, "r", encoding="utf-8") as f:
        # Parse the JSON content back into a Python dictionary.
        d = json.load(f)               
        
        # Append the relevant information to our list.
        # We store the source filename so we know where a retrieved chunk came from.
        chunks.append({
            "text": d["text"],         # The actual text content of the chunk
            "source": os.path.basename(p)   # The filename (e.g., 'case1_chunk0.json')
        })

print("loaded:", len(chunks))

# Extract just the text strings into a separate list.
# This list is what we will pass to the embedding model.
texts = [c["text"] for c in chunks]



# 2. Initialize Embedding Model

print("\nloading bge-base embeddings...")

# Initialize the HuggingFace embedding model via LangChain.
# Model: "BAAI/bge-base-en-v1.5" is a strong, open-source embedding model 
# well-suited for retrieval tasks.
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    # Set device to 'cuda' to utilize the GPU for faster embedding generation.
    model_kwargs={"device": "cuda"},
    encode_kwargs={
        # normalize_embeddings=True ensures all output vectors have a length of 1.
        # This is critical because it allows us to use Inner Product (dot product) 
        # in FAISS to calculate Cosine Similarity.
        "normalize_embeddings": True,
        # Process chunks in batches of 32 to optimize memory and speed.
        "batch_size": 32
    }
)


# 3. Generate Embeddings (Vectors)

print("encoding, wait...")

# Pass the list of texts through the embedding model to generate vectors.
# This step converts human-readable text into high-dimensional numerical representations.
vecs = embedder.embed_documents(texts)

# Convert the resulting list of vectors into a NumPy array.
# FAISS explicitly requires the data type to be float32 for compatibility and performance.
vecs = np.array(vecs).astype("float32")        
print("vector shape:", vecs.shape)



# 4. Build and Save FAISS Index

print("\nbuilding faiss...")

# Initialize a FAISS index.
# IndexFlatIP performs exhaustive search using Inner Product.
# Because our vectors are normalized (magnitude of 1), Inner Product is mathematically 
# equivalent to Cosine Similarity, which is the standard metric for comparing text embeddings.
# vecs.shape[1] is the dimensionality of the vectors (e.g., 768 for bge-base).
index = faiss.IndexFlatIP(vecs.shape[1])

# Add all generated vectors to the FAISS index.
index.add(vecs)        


# Save the FAISS index to disk so it can be loaded instantly during inference/retrieval.
faiss.write_index(index, os.path.join(OUTPUT_DIR, "index.faiss"))

# Save the corresponding metadata (the 'chunks' list) using Python's pickle module.
# The order of the metadata list corresponds exactly to the order of vectors in the FAISS index.
# E.g., The vector at index 5 in FAISS corresponds to chunks[5].
with open(os.path.join(OUTPUT_DIR, "index.pkl"), "wb") as f:
    pickle.dump(chunks, f)


print("\nDONE. saved to:", OUTPUT_DIR)
