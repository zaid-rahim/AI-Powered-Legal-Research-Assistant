import os
import re
from cleantext import clean
from tqdm import tqdm


# Configuration & Directories

# IN_DIR: The directory containing the raw text files extracted from PDFs/Word docs.
IN_DIR  = "data_extracted"      

# OUT_DIR: The directory where the cleaned text files will be saved.
OUT_DIR = "data_clean"          

# Ensure the output directory exists. If it doesn't, os.makedirs creates it.
# exist_ok=True prevents an error if the directory already exists.
os.makedirs(OUT_DIR, exist_ok=True)


# Regular Expression Patterns

# Patterns to identify common headers/footers in legal or formal documents.
# r"..." denotes a raw string.
# \d+ matches one or more digits.
HEADER_PATTERNS = [
    r"Page \d+ of \d+",  # Example: "Page 1 of 10"
    r"Page \d+",         # Example: "Page 2"
]


def remove_header_footer(text):
    """
    Removes lines from the text that match known header/footer patterns.

    Args:
        text (str): The raw text to process.

    Returns:
        str: The text with header/footer lines removed.
    """
    lines = []
    # splitlines() safely splits the string into a list of lines, handling \r\n and \n
    for L in text.splitlines():
        # Check if the current line matches ANY of our header patterns.
        # flags=re.I makes the search case-insensitive (e.g., "PAGE 1" or "page 1").
        if any(re.search(pat, L, flags=re.I) for pat in HEADER_PATTERNS):
            # If it's a match, we skip adding this line to our list.
            continue
        # If it's not a header/footer, add it to our cleaned lines list.
        lines.append(L)
        
    # Rejoin the cleaned lines back into a single string with newline characters.
    return "\n".join(lines)


def clean_text_block(text):
    """
    Applies a series of cleaning operations to standardize the text format.

    Args:
        text (str): The text block to clean.

    Returns:
        str: The fully cleaned and standardized text.
    """
    # 1. Remove page numbers/headers using the custom function above.
    t = remove_header_footer(text)            
    
    # 2. Use the 'cleantext' library to perform aggressive text normalization.
    #    - lower=False: Keeps original capitalization.
    #    - no_urls=True: Replaces URLs with a generic <URL> token (or removes them).
    #    - no_emails=True: Replaces email addresses.
    #    - no_phone_numbers=True: Replaces phone numbers.
    t = clean(t, lower=False, no_urls=True,
              no_emails=True, no_phone_numbers=True)   
              
    # 3. Reduce multiple consecutive blank lines.
    #    re.sub(pattern, replacement, string)
    #    r"\n{3,}" matches 3 or more consecutive newline characters.
    #    We replace them with exactly 2 newlines (one blank line).
    t = re.sub(r"\n{3,}", "\n\n", t)          
    
    # 4. Remove leading and trailing whitespace from the entire document.
    return t.strip()                           



# Main Execution Block

if __name__ == "__main__":
    # Initialize a progress bar over the files in the input directory.
    # tqdm provides a visual loading bar in the console.
    for fname in tqdm(os.listdir(IN_DIR), desc="Cleaning text"):

        # Only process text files; skip images, system files, etc.
        if not fname.endswith(".txt"):
            continue

        # Construct the full file path for reading
        in_path = os.path.join(IN_DIR, fname)
        
        # Read the raw text. Using utf-8 encoding is crucial for special characters.
        with open(in_path, "r", encoding="utf-8") as f:
            txt = f.read()

        # Apply the cleaning pipeline to the text
        cleaned = clean_text_block(txt)

        # Construct the output file path
        out_path = os.path.join(OUT_DIR, fname)
        
        # Write the cleaned text to the new directory
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(cleaned)
