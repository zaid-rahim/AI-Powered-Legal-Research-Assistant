from neo4j import GraphDatabase
import os
import json


# Neo4j Database Configuration

# Define the connection URI and authentication credentials for the Neo4j database.
# bolt:// is the standard protocol for connecting to Neo4j.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Directory where the merged summary JSON files are stored.
SUMMARY_DIR = "training_data/merged_summaries"

# Initialize the Neo4j driver. This object manages connection pools to the database.
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def add_summary(tx, case_id, summary_text):
    """
    Executes a Cypher query to insert a summary and link it to its corresponding case.

    This function is designed to be run within a Neo4j transaction (tx).

    Args:
        tx (neo4j.Transaction): The active Neo4j transaction.
        case_id (str): The unique identifier for the case (e.g., the filename).
        summary_text (str): The merged summary text to store.
    """
    
    # Cypher Query Explanation:
    
    # MERGE (c:Entity {name: $case_id})
    #   -> Finds an existing 'Entity' node with the given case_id, or creates it if it doesn't exist.
    # MERGE (s:Summary {case_id: $case_id})
    #   -> Finds or creates a 'Summary' node specifically for this case_id.
    # SET s.text = $summary_text
    #   -> Updates the 'text' property of the Summary node with our summary_text.
    # MERGE (c)-[:HAS_SUMMARY]->(s)
    #   -> Creates a directed relationship 'HAS_SUMMARY' from the Entity node to the Summary node,
    #      ensuring the link exists between the case and its summary.
    query = """
    MERGE (c:Entity {name: $case_id})
    MERGE (s:Summary {case_id: $case_id})
    SET s.text = $summary_text
    MERGE (c)-[:HAS_SUMMARY]->(s)
    """
    
    # Execute the query, passing in the parameters securely to prevent injection.
    tx.run(query, case_id=case_id, summary_text=summary_text)




# Main Execution Block

if __name__ == "__main__":

    print("Loading merged summaries...")
    
    # Find all JSON files in the summaries directory.
    files = [f for f in os.listdir(SUMMARY_DIR) if f.endswith(".json")]
    print("Found", len(files), "summary files\n")

    # Open a session with the Neo4j driver. A session is a logical context 
    # for executing transactions.
    with driver.session() as session:
        for f in files:

            # Construct the full path and load the JSON data
            path = os.path.join(SUMMARY_DIR, f)
            data = json.load(open(path, "r", encoding="utf-8"))

            # Extract the required fields from the JSON.
            # We expect keys "case_id" and "merged_summary" to exist.
            case_id = data["case_id"]              
            summary_text = data["merged_summary"]  

            print("Inserting summary for:", case_id)

            try:
                # session.execute_write executes the 'add_summary' function 
                # within a write transaction, automatically handling retries 
                # if transient network errors occur.
                session.execute_write(add_summary, case_id, summary_text)
            except Exception as e:
                # Catch and print any database errors (e.g., constraint violations)
                # so the script can continue processing other files.
                print("Error for", case_id, ":", e)

    # Always close the driver when done to release resources cleanly.
    driver.close()
    print("Done inserting all merged summaries into Neo4j.")
