import json
import time
from rag_core import build_vector_store, answer_question

PDF_PATH = "Greenfield_Student_Handbook.pdf"

vector_store = build_vector_store(PDF_PATH)

with open("dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

all_cases = {}

for set_name, rows in dataset.items():
    all_cases[set_name] = []

    for number, row in enumerate(rows, start=1):
        question = row["question"].strip()
        if question == "":
            raise ValueError(f"{set_name} #{number} has an empty question. Fill dataset.json first.")

        print(f"[{set_name} {number}/{len(rows)}] {question}")
        answer, retrieval_context = answer_question(vector_store, question)

        all_cases[set_name].append({
            "input": question,
            "expected_output": row["expected_output"].strip(),
            "actual_output": answer,
            "retrieval_context": retrieval_context,
        })
        time.sleep(1)   # small pause to stay under rate limits

with open("cases.json", "w", encoding="utf-8") as f:
    json.dump(all_cases, f, indent=2, ensure_ascii=False)

print("Saved cases.json")
