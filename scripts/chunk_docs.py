import os
import json
from tqdm import tqdm


# Configuration & Directories

# IN_DIR: Directory where the cleaned text files (.txt) are located.
IN_DIR = "data_clean"                     

# OUT_DIR: Directory where the resulting chunk files (.json) will be saved.
OUT_DIR = "training_data/chunks"          

# Create the output directory if it doesn't already exist.
os.makedirs(OUT_DIR, exist_ok=True)


# Chunking Parameters

# CHUNK_CHARS: The maximum number of characters each text chunk should contain.
# A size of 8000 characters provides a good balance between context size 
# and embedding model limitations.
CHUNK_CHARS = 8000        

# OVERLAP: The number of characters that should overlap between consecutive chunks.
# Overlap is crucial because it prevents sentences or thoughts from being abruptly
# cut off at chunk boundaries, ensuring context is preserved across chunks.
OVERLAP = 500             


def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP):
    """
    Splits a large text document into smaller, overlapping chunks.

    Args:
        text (str): The full text string to be chunked.
        size (int): The maximum number of characters per chunk.
        overlap (int): The number of characters to overlap with the previous chunk.

    Returns:
        list of str: A list containing the text chunks.
    """
    chunks = []
    start = 0
    L = len(text)

    # Loop through the text until our start pointer reaches the end
    while start < L:
        # Determine the end index for the current chunk.
        # It's either start + size, or the end of the text (L) if we're near the end.
        end = min(start + size, L)

        # Extract the chunk from the text and remove leading/trailing whitespace.
        chunk = text[start:end].strip()
        chunks.append(chunk)

        # Update the start pointer for the next chunk.
        # We move it back by the 'overlap' amount to create the overlap.
        # max(..., 0) ensures we don't accidentally get a negative starting index.
        start = max(end - overlap, 0)
        
        # Break condition if we've reached the end to prevent infinite loops
        # in edge cases where end-overlap <= start
        if start >= end:
            break
            
        # If the end index reached the end of the document, we're done.
        if end == L:
            break

    return chunks



# Main Execution Block

if __name__ == "__main__":

    count = 0
    
    # Iterate through all files in the input directory.
    # sorted() ensures we process files in a consistent, alphabetical order.
    # tqdm provides a progress bar in the console.
    for fname in tqdm(sorted(os.listdir(IN_DIR)), desc="Chunking docs"):

        # We only want to process text files.
        if not fname.endswith(".txt"):
            continue
            
        fullpath = os.path.join(IN_DIR, fname)

        # Safely open and read the cleaned text file.
        # utf-8 encoding handles special characters properly.
        with open(fullpath, "r", encoding="utf-8") as f:
            text = f.read()

        # Generate the chunks for the current file.
        chunks = chunk_text(text)

        # Save each chunk as a separate JSON file.
        for i, chunk in enumerate(chunks):

            # Create a dictionary containing metadata and the chunk text.
            # This makes it easier to track which file a chunk came from later
            # (e.g., during vector store creation or retrieval).
            meta = {
                "case_file": fname,
                "chunk_id": i,
                "text": chunk
            }
            
            # Construct a unique filename for the chunk JSON file.
            # Example: "document1.txt" -> "document1chunk0.json", "document1chunk1.json"
            out_path = os.path.join(
                OUT_DIR,
                f"{fname.replace('.txt','')}chunk{i}.json"
            )
            
            # Write the metadata dictionary to a JSON file.
            # ensure_ascii=False allows non-ASCII characters to be saved correctly.
            # indent=2 makes the JSON file human-readable.
            with open(out_path, "w", encoding="utf-8") as out:
                json.dump(meta, out, ensure_ascii=False, indent=2)
                
            count += 1

    # Print a final summary of the operation
    print("Total chunks written:", count)