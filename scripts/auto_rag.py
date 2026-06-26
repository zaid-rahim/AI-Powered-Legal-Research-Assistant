import sys
import os
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Web.backend_wrapper import ask_question
#  PARSER 
def extract_from_rag(text):
    text = text.lower()

    #  Judges 
    judges = re.findall(r'mr\. justice [a-z\s\-]+', text)
    judges = [j.replace("mr. justice", "").strip() for j in judges]

    #  Sections 
    sections_raw = re.findall(r'section\s*[^\n,*]+', text)

    sections = []
    for s in sections_raw:
        s = re.sub(r'\(.*?\)', '', s)   # remove explanations
        s = s.replace("section", "").strip()
        sections.append(s)

    return {
        "judges": "; ".join(sorted(set(judges))),
        "sections": "; ".join(sorted(set(sections)))
    }


#  MAIN LOOP 
def generate_rag_csv():
    data = []

    for i in range(1, 5):
        case_id = f"Crl.A.{i}"
        print(f"\nProcessing {case_id}...")

        try:
            # Ask RAG via your backend
            q1 = f"Who is the judge in {case_id}?"
            q2 = f"What sections are involved in {case_id}?"

            res1 = ask_question(q1)["answer"]
            res2 = ask_question(q2)["answer"]

            parsed1 = extract_from_rag(res1)
            parsed2 = extract_from_rag(res2)

            data.append({
                "case_number": case_id,
                "judges": parsed1["judges"],
                "sections": parsed2["sections"]
            })

        except Exception as e:
            print(f" Error in {case_id}: {e}")
            data.append({
                "case_number": case_id,
                "judges": "",
                "sections": ""
            })

    df = pd.DataFrame(data)
    df.to_csv("rag_output.csv", index=False)

    print("\n rag_output.csv generated!")


#  RUN 
if __name__ == "__main__":
    generate_rag_csv()