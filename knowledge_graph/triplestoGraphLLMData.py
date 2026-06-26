from neo4j import GraphDatabase
import json
import os


# Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USERNAME = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")
BASE_DIR = os.path.dirname(__file__)
INPUT_TRIPLES = os.path.join(BASE_DIR, "cases_triples_with_summary.json")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def create_graph(tx, subject, predicate, obj):

    # Always ensure Case exists
    tx.run("""
        MERGE (c:Case {id: $case_id})
    """, case_id=subject)

    # Ensure Summary node exists (without overwriting)
    tx.run("""
        MERGE (c:Case {id: $case_id})
        MERGE (s:Summary {case_id: $case_id})
        MERGE (c)-[:HAS_SUMMARY]->(s)
    """, case_id=subject)

    # Handle predicates
    if predicate == "HAS_CASE_NO":
        tx.run("""
            MERGE (c:Case {id: $case_id})
            MERGE (n:CaseNumber {value: $value})
            MERGE (c)-[:HAS_CASE_NO]->(n)
        """, case_id=subject, value=obj)

    elif predicate == "HEARD_BY":
        tx.run("""
            MERGE (c:Case {id: $case_id})
            MERGE (j:Judge {name: $value})
            MERGE (c)-[:HEARD_BY]->(j)
        """, case_id=subject, value=obj)

    elif predicate == "HAS_PETITIONER":
        tx.run("""
            MERGE (c:Case {id: $case_id})
            MERGE (p:Petitioner {name: $value})
            MERGE (c)-[:HAS_PETITIONER]->(p)
        """, case_id=subject, value=obj)

    elif predicate == "HAS_RESPONDENT":
        tx.run("""
            MERGE (c:Case {id: $case_id})
            MERGE (r:Respondent {name: $value})
            MERGE (c)-[:HAS_RESPONDENT]->(r)
        """, case_id=subject, value=obj)

    elif predicate == "INVOLVES_SECTION":
        tx.run("""
            MERGE (c:Case {id: $case_id})
            MERGE (s:Section {name: $value})
            MERGE (c)-[:INVOLVES_SECTION]->(s)
        """, case_id=subject, value=obj)

    elif predicate == "HAS_DECISION":
        tx.run("""
            MERGE (c:Case {id: $case_id})
            MERGE (d:Decision {value: $value})
            MERGE (c)-[:HAS_DECISION]->(d)
        """, case_id=subject, value=obj)

    
    elif predicate == "HAS_SUMMARY":
        tx.run("""
            MERGE (c:Case {id: $case_id})
            MERGE (s:Summary {case_id: $case_id})
            SET s.text = $value
            MERGE (c)-[:HAS_SUMMARY]->(s)
        """, case_id=subject, value=obj)

# Load triples
with open(INPUT_TRIPLES, "r", encoding="utf-8") as f:
    triples = json.load(f)

with driver.session(database="legalkg") as session:
    for subject, predicate, obj in triples:
        session.execute_write(create_graph, subject, predicate, obj)


print("Graph successfully created in Neo4j.")
driver.close()