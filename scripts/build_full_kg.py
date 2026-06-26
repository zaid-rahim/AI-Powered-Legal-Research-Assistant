import json
import os
import re
from neo4j import GraphDatabase


# Configuration & Paths

# Neo4j connection details. bolt:// is the standard protocol for Neo4j.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# The JSON file containing extracted relationship triples (Subject, Predicate, Object).
TRIPLES_FILE = "./case_triples.json"     

# The directory containing the finalized merged summaries for the cases.
MERGED_SUMMARY_DIR = "./training_data/merged_summaries"    


# Database Connection

# Initialize the driver that manages connections to the Neo4j instance.
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def normalize_case_id(s):
    """
    Standardizes case ID strings to ensure consistency in the database.
    
    Example: 
    "Criminal Appeal No 123/2010" -> "Crl.A.123"

    Args:
        s (str): The raw case ID string.

    Returns:
        str: The normalized case ID, or the original string if no pattern matches.
    """
    if not s: 
        return None
    s = s.strip()
    
    # Regex to extract the core case number, optionally followed by a year (e.g., /2020)
    m = re.search(r"(\d+)(?:\/\d{4})?$", s)
    
    # Check if the string indicates a criminal appeal.
    if m and ("Crl" in s or "Criminal" in s or s.lower().startswith("crl.a")):
        return "Crl.A." + m.group(1)
        
    return s     # Fallback: keep original if pattern doesn't match


def clean_text(t):
    """
    Ensures the input is a stripped string. Prevents NoneType errors.
    """
    if t is None: 
        return None
    return str(t).strip()



# Cypher Queries Dictionary

# Pre-defined Cypher queries for creating nodes and relationships.
# Using MERGE ensures we don't create duplicate nodes if they already exist.
CREATE_QUERIES = {
    "case": """
        MERGE (c:Case {case_id: $case_id})
        ON CREATE SET c.created = timestamp()
        RETURN c.case_id
    """,
    "person": """
        MERGE (p:Person {name: $name})
        RETURN p.name
    """,
    "judge": """
        MERGE (j:Judge {name: $name})
        RETURN j.name
    """,
    "fir": """
        MERGE (f:FIR {number: $number})
        ON CREATE SET f.date = $date
        RETURN f.number
    """,
    "section": """
        MERGE (s:Section {code: $code})
        RETURN s.code
    """,
    # Note: %s is used for dynamic relationship types (labels), as Neo4j parameters ($var) 
    # cannot be used for relationship types or node labels.
    "rel_case_person": """
        MERGE (c:Case {case_id: $case_id})
        MERGE (p:Person {name: $person_name})
        MERGE (c)-[r:%s]->(p)
        RETURN type(r)
    """,
    "rel_case_judge": """
        MERGE (c:Case {case_id: $case_id})
        MERGE (j:Judge {name: $judge_name})
        MERGE (c)-[r:%s]->(j)
        RETURN type(r)
    """,
    "rel_case_fir": """
        MERGE (c:Case {case_id: $case_id})
        MERGE (f:FIR {number: $fir_number})
        SET f.date = coalesce($fir_date, f.date)
        MERGE (c)-[r:%s]->(f)
        RETURN type(r)
    """,
    "rel_case_section": """
        MERGE (c:Case {case_id: $case_id})
        MERGE (s:Section {code: $code})
        MERGE (c)-[r:%s]->(s)
        RETURN type(r)
    """
}


# Predicate Mapping

# Maps raw string predicates extracted from text (e.g., "hasJudge") 
# to a standardized Neo4j relationship label ("HAS_JUDGE") and the target node type ("judge").
REL_MAPPING = {
    "hasJudge": ("HAS_JUDGE", "judge"),
    "hasjudge": ("HAS_JUDGE", "judge"),
    "hasPetitioner": ("HAS_PETITIONER", "person"),
    "hasRespondent": ("HAS_RESPONDENT", "person"),
    "hasComplainant": ("HAS_COMPLAINANT", "person"),
    "hasAccused": ("HAS_ACCUSED", "person"),
    "hasWitness": ("HAS_WITNESS", "person"),
    "hasFIR": ("HAS_FIR", "fir"),
    "hasSection": ("REFERS_TO_SECTION", "section"),
    "refersToLaw": ("REFERS_TO_SECTION", "section"),
    "decidedIn": ("DECIDED_IN", "person"),
    "decision": ("DECISION", "person"),
    "hasSummary": ("HAS_SUMMARY", "summary"),
}


def map_pred(pred):
    """
    Converts a raw predicate string into its corresponding Neo4j relationship label and node type.
    """
    if not pred:
        return ("RELATED_TO", "person")
        
    key = pred.strip().replace(" ", "")
    # Default to generic RELATED_TO and person if the predicate isn't in the mapping.
    return REL_MAPPING.get(key, ("RELATED_TO", "person"))


def run_create(session, cypher, **params):
    """
    Helper function to execute a Cypher query safely and catch exceptions.
    """
    try:
        session.run(cypher, **params)
    except Exception as e:
        print("   Error running cypher:", e, "params:", params)



# Graph Construction Functions

def import_triples(triples):
    """
    Iterates through a list of (Subject, Predicate, Object) triples 
    and inserts them into the Neo4j database as nodes and relationships.
    """
    print("Importing triples:", len(triples))

    # Open a session with the database.
    with driver.session() as session:
        for subj, pred, obj in triples:

            # Clean and normalize the data
            subj_clean = clean_text(subj)
            obj_clean = clean_text(obj)
            pred_clean = clean_text(pred)

            subj_case = normalize_case_id(subj_clean)

            # Rule 1: The subject is a recognized legal case.
            if subj_case and subj_case.startswith("Crl.A"):
                
                # Ensure the root Case node exists.
                run_create(session, CREATE_QUERIES["case"], case_id=subj_case)

                # Determine the type of relationship and the type of the object node.
                rel_label, target_type = map_pred(pred_clean)

                # Route the creation logic based on the target node type.
                if target_type == "judge":
                    if obj_clean:
                        run_create(session, CREATE_QUERIES["judge"], name=obj_clean)
                        # Inject the dynamic relationship label into the query string
                        cy = CREATE_QUERIES["rel_case_judge"] % rel_label
                        run_create(session, cy, case_id=subj_case, judge_name=obj_clean)

                elif target_type == "person":
                    if obj_clean:
                        run_create(session, CREATE_QUERIES["person"], name=obj_clean)
                        cy = CREATE_QUERIES["rel_case_person"] % rel_label
                        run_create(session, cy, case_id=subj_case, person_name=obj_clean)

                elif target_type == "fir":
                    # FIR numbers sometimes contain dates. Try to extract it.
                    fir_num = obj_clean
                    m = re.search(r"(\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4})", obj_clean)
                    fir_date = m.group(1) if m else None
                    
                    cy = CREATE_QUERIES["rel_case_fir"] % rel_label
                    run_create(session, cy, case_id=subj_case, fir_number=fir_num, fir_date=fir_date)

                elif target_type == "section":
                    cy = CREATE_QUERIES["rel_case_section"] % rel_label
                    run_create(session, cy, case_id=subj_case, code=obj_clean)

                else:
                    # Fallback: Create a generic Entity node and RELATED_TO relationship.
                    run_create(session, """
                        MERGE (c:Case {case_id: $case_id})
                        MERGE (e:Entity {name: $obj})
                        MERGE (c)-[:RELATED_TO]->(e)
                    """, case_id=subj_case, obj=obj_clean)

            # Rule 2: The subject is NOT a case (e.g., a person related to another person).
            else:
                run_create(session, "MERGE (e:Entity {name: $name})", name=subj_clean)



def import_summaries(summary_dir):
    """
    Reads the merged summary JSON files and attaches the text to the 
    corresponding Case nodes in the Knowledge Graph.
    """
    print("Importing merged summaries from:", summary_dir)

    files = [f for f in os.listdir(summary_dir) if f.endswith(".json")]

    with driver.session() as session:
        for fname in files:
            path = os.path.join(summary_dir, fname)

            try:
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception as e:
                print("   Could not read file:", path, e)
                continue

            # Extract and normalize the case ID.
            case_id = data.get("case_id") or os.path.splitext(fname)[0]
            case_id = normalize_case_id(case_id) or case_id

            # Extract the summary text, checking multiple possible keys.
            summary_text = (
                data.get("merged_summary") or
                data.get("summary") or
                data.get("text")
            )

            if not summary_text:
                print("   No summary text for", case_id)
                continue

            # Create a Summary node and link it to the Case node.
            session.run("""
                MERGE (c:Case {case_id: $case_id})
                MERGE (s:Summary {case_id: $case_id})
                SET s.text = $text
                MERGE (c)-[:HAS_SUMMARY]->(s)
            """, case_id=case_id, text=summary_text)

            print("   Added summary for", case_id)



# Main Execution Block

if __name__ == "__main__":

    print("Building full knowledge graph...")

    # 1. Load and process Triples
    triples = []
    if os.path.exists(TRIPLES_FILE):
        with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
            triples = json.load(f)
    else:
        print("Triples file missing:", TRIPLES_FILE)

    if triples:
        import_triples(triples)
    else:
        print("No triples found, skipping triple import.")

    # 2. Load and process Summaries
    if os.path.isdir(MERGED_SUMMARY_DIR):
        import_summaries(MERGED_SUMMARY_DIR)
    else:
        print("Summary folder missing:", MERGED_SUMMARY_DIR)

    # Clean up connections
    driver.close()
    print("Done. KG built.")
