import os
import glob
import json
import pickle
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


# Configuration & Paths

# MERGED_DIR: The directory where the finalized, merged case summaries are stored.
MERGED_DIR = r"C:\Users\dauda\OneDrive\Desktop\LLAMA_4_LEGAL_V2_REAL\training_data\merged_summaries"

# OUTPUT_DIR: The directory where the FAISS index for the summaries will be saved.
OUTPUT_DIR = r"C:\Users\dauda\OneDrive\Desktop\LLAMA_4_LEGAL_V2_REAL\vector_store_summaries"

# Create the output directory if it doesn't already exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading merged summaries...")


# 1. Load Summary Data

# Find all JSON files in the merged summaries directory.
summary_files = glob.glob(os.path.join(MERGED_DIR, "*.json"))
print("Found:", len(summary_files))

# Initialize a list to hold dictionaries containing the case ID and the summary text.
documents = []      

# Iterate through each summary file.
for path in summary_files:
    with open(path, "r", encoding="utf-8") as f:
        # Parse the JSON file into a Python dictionary.
        data = json.load(f)

    # Extract the case_id. If it's missing from the JSON, fallback to using the filename.
    case_id = data.get("case_id", os.path.basename(path).replace(".json", ""))
    
    # Extract the actual merged summary text. Default to empty string if missing.
    summary = data.get("merged_summary", "")

    # Append the extracted data to our documents list.
    documents.append({
        "case_id": case_id,
        "summary": summary
    })

print("Loaded:", len(documents))

# Create a flat list containing only the summary texts.
# This list will be fed into the embedding model.
texts = [doc["summary"] for doc in documents]



# 2. Initialize Embedding Model

print("\nLoading BGE-base embeddings model...")

# Initialize the HuggingFace embedding model (BGE-base-en-v1.5).
# BGE is currently one of the top-performing open-source embedding models.
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    # Set device to 'cuda' to utilize GPU acceleration, drastically speeding up embedding.
    model_kwargs={"device": "cuda"},
    encode_kwargs={
        # normalize_embeddings=True is required so we can use Inner Product (IP) 
        # in FAISS as a substitute for Cosine Similarity.
        "normalize_embeddings": True,
        # Process 16 documents at a time.
        "batch_size": 16
    }
)



# 3. Generate Embeddings (Vectors)

print("\nEncoding merged summaries...")

# Convert the list of summary texts into high-dimensional numerical vectors.
vectors = embedder.embed_documents(texts)

# Convert the resulting Python list of vectors into a NumPy array.
# FAISS strictly requires float32 data types.
vectors = np.array(vectors).astype("float32")    

print("Vector shape:", vectors.shape)



# 4. Build and Save FAISS Index

print("\nBuilding FAISS index...")

# Initialize a FAISS index using Inner Product (IndexFlatIP).
# Since our vectors are normalized, Inner Product calculates the exact Cosine Similarity.
# vectors.shape[1] dynamically gets the vector dimensionality (e.g., 768).
index = faiss.IndexFlatIP(vectors.shape[1])   

# Add the generated vectors into the FAISS index.
index.add(vectors)                             

# Save the populated FAISS index to the specified output directory.
faiss.write_index(index, os.path.join(OUTPUT_DIR, "index.faiss"))

# Save the original metadata (case IDs and summary text) using pickle.
# This allows us to map the FAISS search results (which only return indices) 
# back to the actual text and case ID.
with open(os.path.join(OUTPUT_DIR, "index.pkl"), "wb") as f:
    pickle.dump(documents, f)

print("\nDone. Summary embeddings saved to:", OUTPUT_DIR)
