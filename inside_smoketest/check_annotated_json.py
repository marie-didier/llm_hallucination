import json
from collections import Counter

path = "qa_generations_820_12_annotated.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("num_items:", len(data))

bad_len = 0
missing_fields = 0
label_values = Counter()

for i, item in enumerate(data):
    required = ["id", "question", "gold_answer", "generations", "hallucination_labels"]
    if any(k not in item for k in required):
        missing_fields += 1
        continue

    gens = item["generations"]
    labels = item["hallucination_labels"]

    if len(gens) != len(labels):
        bad_len += 1

    for x in labels:
        label_values[x] += 1

print("items_with_missing_fields:", missing_fields)
print("items_with_len_mismatch:", bad_len)
print("label_values:", dict(label_values))

print("\nfirst_item_keys:", list(data[0].keys()))
print("first_item_num_generations_field:", data[0].get("num_generations"))
print("first_item_len_generations:", len(data[0]["generations"]))
print("first_item_len_labels:", len(data[0]["hallucination_labels"]))
