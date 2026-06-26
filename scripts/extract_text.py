import os
import pdfplumber
import pytesseract
from PIL import Image
from tqdm import tqdm


# Configuration & Directories

# INPUT_DIR: The directory containing the original raw PDF files.
INPUT_DIR = "data_raw_pdfs"

# OUTPUT_DIR: The directory where the extracted text files will be saved.
OUTPUT_DIR = "data_extracted"

# Create the output directory if it doesn't already exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)


# OCR Configuration

# Configure the path to the Tesseract executable. 
# Tesseract is used for Optical Character Recognition (OCR) to extract text 
# from scanned images inside the PDF when regular text extraction fails.
# Ensure this path matches the installation directory on your system.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image_page(pil_image):
    """
    Performs Optical Character Recognition (OCR) on an image using Tesseract.

    Args:
        pil_image (PIL.Image): An image object representing a single PDF page.

    Returns:
        str: The text extracted from the image.
    """
    # lang='eng' specifies that we are expecting English text.
    return pytesseract.image_to_string(pil_image, lang='eng')


def extract_pdf_text(pdf_path):
    """
    Extracts text from a given PDF file.
    
    It employs a hybrid approach:
    1. First, it attempts to extract regular, selectable text using pdfplumber.
    2. If a page contains very little or no selectable text (e.g., it's a scanned document),
       it falls back to rendering the page as an image and running OCR on it.

    Args:
        pdf_path (str): The full path to the PDF file.

    Returns:
        str: The full text extracted from all pages of the PDF.
    """
    parts = []           # List to store the extracted text for each page

    # Open the PDF file safely using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through every page in the document
        for p in pdf.pages:

            # Attempt 1: Try to extract native selectable text
            page_text = p.extract_text()      

            # Check if we successfully extracted a meaningful amount of text.
            # > 50 characters is a heuristic to check if the page has real text 
            # rather than just a stray character or watermark.
            if page_text and len(page_text.strip()) > 50:
                parts.append(page_text)
            else:
                # Attempt 2: Fallback to OCR for scanned pages or images
                try:
                    # Convert the PDF page to a high-resolution PIL Image.
                    # resolution=300 (DPI) is generally the standard for good OCR results.
                    pil_img = p.to_image(resolution=300).original
                    
                    # Run the OCR function on the image
                    txt = ocr_image_page(pil_img)
                    parts.append(txt)
                except Exception as e:
                    # If both native extraction and OCR fail (e.g., corrupt page),
                    # we append an empty string to maintain page flow without crashing.
                    parts.append("")

    # Join all the extracted page texts together, separated by double newlines.
    return "\n\n".join(parts)



# Main Execution Block

if __name__ == "__main__":

    # Get a list of all PDF files in the input directory.
    # f.lower().endswith(".pdf") ensures we catch .PDF as well as .pdf files.
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    # Loop through the list of PDF files, displaying a progress bar.
    for f in tqdm(files, desc="Extracting PDF text"):

        pdf_path = os.path.join(INPUT_DIR, f)

        try:
            # Perform the hybrid extraction on the current PDF
            text = extract_pdf_text(pdf_path)

            # Define the output file path, changing the extension from .pdf to .txt
            out_path = os.path.join(OUTPUT_DIR, f.replace(".pdf", ".txt"))
            
            # Write the extracted text to the output file.
            # utf-8 encoding is essential for handling special characters correctly.
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(text)

        except Exception as e:
            # If the entire file fails to process (e.g., file not found, permission error),
            # print the error but allow the loop to continue with the next file.
            print("ERROR processing", f, e)
