import os
import json
import time
import requests
from tqdm import tqdm


# Configuration & Directories

# CHUNKS_DIR: Directory containing the input text chunks (.json).
CHUNKS_DIR = "training_data/chunks"

# OUT_DIR: Directory where the successfully generated summaries will be saved.
OUT_DIR    = "training_data/summaries"

# FAILED_DIR: Directory to store metadata for chunks that failed to summarize.
FAILED_DIR = "training_data/failed_summaries"

# Ensure necessary directories exist before starting.
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)


def summarize_chunk(text, retries=5):
    """
    Sends a text chunk to a local LLaMA model (via Ollama) to generate a summary.

    This function includes a retry mechanism to handle timeouts, empty responses, 
    or poor-quality generations.

    Args:
        text (str): The text chunk to be summarized.
        retries (int): The maximum number of attempts to generate a valid summary.

    Returns:
        str or None: The generated summary text, or None if all retries failed.
    """
    # Define the instruction prompt for the LLM. 
    # This guides the model on the format and content expected.
    prompt = (
        "Summarize the following court judgment CHUNK in 6–9 lines.\n"
        "The summary should include: case background, parties, key facts,\n"
        "issues, arguments, reasoning, final decision, citations.\n\n"
        "TEXT:\n"
        f"{text}\n\n"
        "SUMMARY:\n"
    )

    # Define the JSON payload to send to the Ollama REST API.
    payload = {
        "model": "llama3.1:8b",   # Specify the model to use
        "prompt": prompt,         # The composed prompt
        "stream": False,          # We want the complete response at once, not streamed
        "options": {
            "temperature": 0.1,   # Low temperature = more deterministic, factual output
            "num_predict": 450    # Limit the output length to ~450 tokens
        }
    }

    # URL for the local Ollama API generation endpoint
    url = "http://localhost:11434/api/generate"

    # Retry loop
    for attempt in range(retries):
        try:
            # Send the POST request to the Ollama API. 
            # timeout=300 (5 minutes) prevents the script from hanging indefinitely 
            # if the model gets stuck.
            r = requests.post(url, json=payload, timeout=300)
            data = r.json()

            # Check if the API returned a valid response structure.
            if "response" not in data:
                print("Empty response, retry", attempt + 1)
                time.sleep(2)  # Wait before retrying
                continue

            # Extract the generated summary and remove surrounding whitespace
            summary = data["response"].strip()

            # Quality check: If the summary is extremely short (less than 30 chars), 
            # it's likely a failure or an error message from the model.
            if len(summary) < 30:
                print("Bad summary, retry", attempt + 1)
                time.sleep(3)
                continue

            # If we reach here, we have a valid summary.
            return summary

        except Exception as e:
            # Catch network errors, timeouts, or JSON parsing errors.
            print("Error attempt", attempt + 1, ":", e)
            time.sleep(3)

    # If the loop finishes without returning, all retries failed.
    return None   



# Main Execution Block

if __name__ == "__main__":

    # Get a sorted list of all JSON chunk files.
    # Sorting ensures they are processed in order (chunk_0, chunk_1, etc.)
    files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")])

    print("LLaMA summarizing... total chunks:", len(files))

    # Iterate through the files with a progress bar (tqdm).
    for fname in tqdm(files, desc="Summaries"):

        path = os.path.join(CHUNKS_DIR, fname)
        
        # Load the chunk data
        data = json.load(open(path, "r", encoding="utf-8"))

        # Extract the text. We cap it at 20,000 characters to prevent 
        # exceeding the LLM's context window size, which would cause an error.
        text = data["text"][:20000]   

        # Call the summarization function
        summary = summarize_chunk(text)

        # Handle Failed Summaries
        if summary is None:
            fail_path = os.path.join(FAILED_DIR, fname)
            
            # Save the failed text to the failed directory so it can be reviewed 
            # or retried manually later.
            json.dump(
                {"case_id": fname.replace(".json", ""), "text": text},
                open(fail_path, "w", encoding="utf-8"),
                indent=2, ensure_ascii=False
            )
            print("FAILED:", fname)
            continue

        # Handle Successful Summaries
        out_path = os.path.join(OUT_DIR, fname)
        
        # Save the generated summary to the output directory.
        json.dump(
            {
                "case_id": fname.replace(".json", ""),
                "summary": summary
            },
            open(out_path, "w", encoding="utf-8"),
            indent=2, ensure_ascii=False
        )

    print("DONE. All summaries saved in:", OUT_DIR)
