import pickle
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-num', '--numSamples', type=int)
parser.add_argument('-numGen', '--numGenerations', type=int)

args = parser.parse_args()
input_file = args.filename

with open(input_file, 'rb') as f:
    data = pickle.load(f)

# Extract first N samples
n_samples = args.numSamples or len(data)
n_gen = args.numGenerations or int(sum(len(d['generations']) for d in data[:100])/100)

output = []

for i, item in enumerate(data[:n_samples]):
    output.append({
        "id": item["id"],
        "question": item["question"],
        "gold_answer": item["answer"],
        "num_generations": len(item["generations"]),
        "generations": item["generations"][:n_gen],  # first n_gen generations
        "most_likely_generation": item["most_likely_generation"]
    })

# Save JSON
with open(f'qa_generations_{n_samples}_{n_gen}.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

print(f"Saved {len(output)} samples in qa_generations_{n_samples}_{n_gen}.json")

# Stats 
print(f"\nTotal samples: {len(data)}")
print(f"Average generation per sample: {sum(len(d['generations']) for d in data[:100])/100:.1f}")
