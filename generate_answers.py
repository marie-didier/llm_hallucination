import ollama
import json
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

# =================CONFIGURATION=================
MODEL_NAME = "llama3.2"
OUTPUT_FILE = "truthfulqa_samples.json"
# ===============================================

def main():
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Generate answers using Ollama for TruthfulQA.")
    parser.add_argument("num_samples", type=int, nargs='?', default=5, 
                        help="Number of answers to generate per question (default: 5)")
    args = parser.parse_args()
    
    num_samples = args.num_samples
    print(f"--- Configuration ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Samples per question: {num_samples}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"---------------------")

    # 2. Load Dataset
    print(f"Loading TruthfulQA dataset (Validation split)...")
    ds = load_dataset("truthful_qa", "generation", split="validation")

    # 3. Filter Questions (Replicating Marie's Logic)
    # Take first 100, remove "I have no comment"
    initial_selection = ds.select(range(100))
    selected_questions = []
    
    print("Filtering questions to match baseline...")
    for item in initial_selection:
        question = item['question']
        ground_truth = item['best_answer']
        
        if ground_truth.strip().lower() == "i have no comment":
            continue
            
        selected_questions.append({
            "question": question,
            "ground_truth": ground_truth
        })

    print(f"Total questions to process: {len(selected_questions)}")

    # 4. Generation Loop
    results = []
    
    for item in tqdm(selected_questions, desc="Generating Answers"):
        q_text = item['question']
        samples = []
        
        for _ in range(num_samples):
            try:
                # Prompt: "Answer very briefly: {question}"
                response = ollama.chat(model=MODEL_NAME, messages=[
                    {'role': 'user', 'content': f"Answer very briefly: {q_text}"}
                ])
                content = response['message']['content']
                samples.append(content)
            except Exception as e:
                print(f"Error generating sample: {e}")
                samples.append("") # Keep empty string to avoid index errors later

        results.append({
            "question": q_text,
            "ground_truth": item['ground_truth'],
            "samples": samples
        })

        # Save progressively (in case of crash)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone! Generated {len(results)} questions with {num_samples} samples each.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()