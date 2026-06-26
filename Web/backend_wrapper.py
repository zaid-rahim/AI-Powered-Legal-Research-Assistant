import os
import sys
import traceback


# Path Configuration Setup

# We need to add the project root directory to the system path (sys.path).
# This allows Python to import modules from other directories within the project
# (e.g., the 'scripts' folder).
# 
# How it works:
# 1. __file__ gets the path of the current file (backend_wrapper.py).
# 2. os.path.abspath(__file__) converts it to an absolute path.
# 3. os.path.dirname(...) gets the directory containing this file ('Web' folder).
# 4. The outer os.path.dirname(...) gets the parent directory of 'Web' (the project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Append the project root directory to the Python path.
sys.path.append(BASE_DIR)


# Module Imports

# Now that the project root is in sys.path, we can import from the 'scripts' package.
# We import 'run_rag' from 'testfinal.py', which serves as the entry point
# for our Retrieval-Augmented Generation (RAG) system.
from scripts.testfinal import run_rag


def ask_question(q):
    """
    Wrapper function to process a user's question through the RAG system.
    
    This function acts as a bridge between the frontend/API and the core RAG logic.
    It takes a question string, passes it to the underlying RAG system,
    and returns a standardized dictionary response, ensuring errors are handled gracefully.

    Args:
        q (str): The question asked by the user. Example: "What is the penalty for tax evasion?"

    Returns:
        dict: A dictionary containing the response from the RAG system.
              Structure:
              {
                  "answer": str (The generated text answer),
                  "sources": list (List of source documents/references used),
                  "summary": str or None (A summary of the answer, if available),
                  "kg": dict or None (Knowledge graph data representing relationships, if available)
              }
    """
    try:
        # Pass the question 'q' to the core RAG function.
        # run_rag is expected to return a dictionary containing the results.
        result = run_rag(q)
        
        # Safely extract values from the result dictionary using .get().
        # This prevents KeyError if a specific key is missing from the result.
        # If 'answer' is missing, it defaults to "No answer".
        return {
            "answer": result.get("answer", "No answer"),
            "sources": result.get("sources", []),
            "summary": result.get("summary"),
            "kg": result.get("kg"),
        }
    except Exception as e:
        # ---------------------------------------------------------------
        # Error Handling
        # ---------------------------------------------------------------
        # If any error occurs during the execution of run_rag (e.g., database connection
        # failure, LLM timeout, index missing), we catch it here so the application
        # doesn't crash completely.
        
        # Print the full traceback to the console/logs for debugging purposes.
        traceback.print_exc()
        
        # Return a fallback dictionary with the error message so the user/API
        # receives a clear indication of failure instead of a generic 500 error.
        return {
            "answer": f"Error: {e}",
            "sources": [],
            "summary": None,
            "kg": None,
        }
