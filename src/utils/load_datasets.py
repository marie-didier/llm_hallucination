"""
Dataset Standardization and Loading Script
------------------------------------------
This script provides functions to load and unify TriviaQA and TruthfulQA 
datasets into a consistent format. 

Unified Format:
{
    "question": str,
    "ground_truth": str,
    "responses": [
        {"text": str, "is_hallucination": int},
        ...
    ]
}
"""

import json
import os
CURRENT_SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_SCRIPT_PATH)))
def load_triviaqa(file_path):
    """
    Standardizes TriviaQA data by pairing generations with their hallucination labels.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TriviaQA file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    standardized = []
    for item in data:
        responses = []
        # TriviaQA stores text and labels in two parallel lists
        # We zip them to create a list of objects
        for text, label in zip(item['generations'], item['hallucination_labels']):
            responses.append({
                "text": text.strip(),
                "is_hallucination": label
            })
        
        standardized.append({
            "question": item['question'],
            "ground_truth": item['gold_answer'],
            "responses": responses
        })
    return standardized

def load_truthfulqa(file_path):
    """
    Standardizes TruthfulQA data by mapping internal response keys.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TruthfulQA file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    standardized = []
    for item in data:
        responses = []
        # TruthfulQA already has a list of objects, we just ensure key naming consistency
        for resp in item['model_responses']:
            responses.append({
                "text": resp['text'].strip(),
                "is_hallucination": resp['is_hallucination']
            })
            
        standardized.append({
            "question": item['question'],
            "ground_truth": item['ground_truth_reference'],
            "responses": responses
        })
    return standardized

def load_all_datasets(trivia_path, truthful_path):
    """
    Loads both datasets and returns them in a single dictionary.
    """
    print("Starting dataset loading and standardization...")
    
    datasets = {
        "triviaqa": load_triviaqa(trivia_path),
        "truthfulqa": load_truthfulqa(truthful_path)
    }
    
    print(f"Successfully loaded {len(datasets['triviaqa'])} items from TriviaQA.")
    print(f"Successfully loaded {len(datasets['truthfulqa'])} items from TruthfulQA.")
    
    return datasets

if __name__ == "__main__":
    # Define your local paths
    TRIVIA_PATH = "../../data/triviaqa/qa_generations_820_12_annotated.json"
    TRUTHFUL_PATH = "../../data/truthfulqa/truthfulqa_with_hallucination_truth.json"
    
    try:
        # Load and unify everything
        all_data = load_all_datasets(TRIVIA_PATH, TRUTHFUL_PATH)
        
        # Verification check: Print the first item of each dataset
        print("\n--- TriviaQA Standardized Sample ---")
        print(json.dumps(all_data['triviaqa'][0], indent=2))
        
        print("\n--- TruthfulQA Standardized Sample ---")
        print(json.dumps(all_data['truthfulqa'][0], indent=2))
        
    except Exception as e:
        print(f"An error occurred: {e}")