import os
from neo4j import GraphDatabase
import pandas as pd

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
username = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(uri, auth=(username, password))

def get_case_data(tx, case_number):
    query = """
    MATCH (c:Case {id: $case_number})

    OPTIONAL MATCH (c)-[:HEARD_BY]->(j:Judge)
    OPTIONAL MATCH (c)-[:INVOLVES_SECTION]->(s:Section)

    RETURN 
        $case_number AS case_number,
        collect(DISTINCT j.name) AS judges,
        collect(DISTINCT s.name) AS sections
    """
    
    result = tx.run(query, case_number=case_number)
    return [record.data() for record in result]

data = []

with driver.session(database="legalkg") as session:
    for i in range(1, 301):
        case_id = f"Crl.A.{i}"
        
        result = session.execute_read(get_case_data, case_id)
        
        record = result[0]
        
        data.append({
            "case_number": record["case_number"],
            "judges": "; ".join(sorted(set([j for j in record["judges"] if j]))),
            "sections": "; ".join(sorted(set([s for s in record["sections"] if s])))
        })

df = pd.DataFrame(data)
df.to_csv("ground_truth.csv", index=False)

print(" CSV generated")
driver.close()