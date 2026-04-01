# LLM Hallucination Detection

Detect hallucinations in LLM responses using **Kernel Language Entropy** and NLI models.

## About

This project implements uncertainty quantification methods to detect hallucinations in LLM-generated text using:
- **Kernel Language Entropy** (via [lm-polygraph](https://github.com/IINemo/lm-polygraph))
- **NLI models** for semantic similarity between responses

## Installation

```bash
# Clone the repository
git clone https://github.com/marie-didier/llm-hallucination.git
cd llm-hallucination

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

## Running tests

### NLI Computing

From the root directory, run:
```bash
python -m src.tests.run_nli
```

### KLE Scores

From the root directory, run:
```bash
python -m src.tests.run_kle
```