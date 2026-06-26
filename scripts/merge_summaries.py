import os
import json
from collections import defaultdict


# Configuration & Directories

# SUMMARIES_DIR: Directory where the individual summaries for each chunk are stored.
SUMMARIES_DIR = "training_data/summaries"

# OUT_DIR: Directory where the final, merged summary for each complete case will go.
OUT_DIR = "training_data/merged_summaries"

# Create the output directory if it doesn't already exist.
os.makedirs(OUT_DIR, exist_ok=True)

print("Scanning summaries folder...")


# 1. Group Chunk Summaries by Case

# Initialize a defaultdict where each key will be a case_id, 
# and the value will be a list of filenames belonging to that case.
# Using defaultdict(list) prevents KeyError when appending to a new key.
case_groups = defaultdict(list)

# Loop through all files in the summaries directory.
# sorted() ensures the chunks are processed in order (e.g., chunk_0, chunk_1).
for fname in sorted(os.listdir(SUMMARIES_DIR)):
    if not fname.endswith(".json"):
        continue

    # Filename format example: "Crl.A.100_chunk_0.json"
    # We split by "_chunk_" and take the first part to get the base case_id.
    # So "Crl.A.100_chunk_0.json" -> "Crl.A.100"
    case_id = fname.split("_chunk_")[0]
    
    # Add this specific chunk filename to the list for this case.
    case_groups[case_id].append(fname)

print("Found", len(case_groups), "cases to merge\n")


def merge_case(case_id, files):
    """
    Reads all individual chunk summaries for a specific case, merges them 
    into a single text string, and saves the result as a new JSON file.

    Args:
        case_id (str): The identifier for the case (e.g., "Crl.A.100").
        files (list): A list of filenames containing the chunk summaries for this case.

    Returns:
        str: The path to the newly created merged summary file.
    """
    # Initialize the data structure that will be saved as the merged JSON file.
    merged = {
        "case_id": case_id,
        "merged_summary": "",      # Will hold the combined text
        "chunk_summaries": []      # Will hold the individual chunk summaries for reference
    }

    # A temporary list to collect just the text strings of the summaries
    # to make joining them together easier later.
    all_summaries = []      

    # Iterate through each chunk file for this case
    for fname in files:
        path = os.path.join(SUMMARIES_DIR, fname)
        
        # Open and parse the JSON file
        data = json.load(open(path, encoding="utf-8"))

        # Extract the summary text. 
        # .get() prevents errors if the key is missing. .strip() removes whitespace.
        summary = data.get("summary", "").strip()
        
        if summary:
            # Add the text to our temporary list for merging
            all_summaries.append(summary)

        # Also store the individual summary in the metadata section
        merged["chunk_summaries"].append({
            "chunk_file": fname,
            "summary": summary
        })

    # Combine all the individual chunk summaries into one long string.
    # We use "\n\n" (double newline) to separate the chunks cleanly.
    merged["merged_summary"] = "\n\n".join(all_summaries)

    
    # 2. Save Merged Summary
    
    # Construct the output filename (e.g., "Crl.A.100.json")
    out_path = os.path.join(OUT_DIR, f"{case_id}.json")
    
    # Write the complete dictionary to the output file.
    # indent=2 makes the JSON easily readable by humans.
    # ensure_ascii=False ensures special characters are saved correctly.
    with open(out_path, "w", encoding="utf-8") as outfile:
        json.dump(merged, outfile, indent=2, ensure_ascii=False)

    return out_path




# Main Execution Block

print("Merging all case summaries...")

# Iterate through the dictionary we built earlier.
# case_id is the key, files is the list of chunk filenames.
for case_id, files in case_groups.items():
    merge_case(case_id, files)

print("\nDONE! Merged summaries saved to:", OUT_DIR)
