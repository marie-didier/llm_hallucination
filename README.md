# INSIDE Hallucination Detection for LLMs

Short implementation and evaluation of INSIDE-EigenScore for question-level hallucination risk detection, with feature clipping kept as an exploratory extension.

## 1. Scope

This repository implements a practical INSIDE-based pipeline for hallucination risk detection in factual question answering.

Implemented in this repository:
- INSIDE-EigenScore
- real hidden states from the model
- own generations for scoring
- question-level evaluation
- final tables and figures for the project report

Kept as exploratory:
- feature clipping

Out of scope:
- full reproduction of the INSIDE paper
- exhaustive multi-model comparison
- feature clipping as a final replacement for the baseline

## 2. Main contribution

This repository provides:
- a working INSIDE-EigenScore baseline
- evaluation on QA820 and TriviaQA
- scaling analysis on QA820
- an exploratory feature clipping branch
- final artifacts for reporting in `results/final/`

## 3. Repository structure

- `src/inside/` core implementation
- `configs/` official experiment configs
- `scripts/` experiment and reporting scripts
- `data/raw/` input datasets
- `data/audit/` dataset audit outputs
- `results/local_runs/` raw run outputs
- `results/final/` final summaries, tables, figures, and cases
- `legacy/` old or unused material

## 4. Setup

Recommended:
- Python 3.10+
- CUDA GPU
- access to the Hugging Face model if required

Create the environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the model requires authentication:
```bash
huggingface-cli login
```

## 5. Data

Expected files in `data/raw/`:
- `qa_generations_820_12_annotated.json`
- `triviaqa_annotated.json`

Main usage:
- `question` is used as model input
- `gold_answer` is used as reference
- stored generations in the JSON files are not used to compute EigenScore
- if `num_generations` is inconsistent, the code uses the real list length

Dataset audit:
```bash
python scripts/audit_dataset.py
```

## 6. Official experiments

Baseline:
- smoke
- K ablation
- QA820 with `n = 100, 300, 500, 820`
- TriviaQA with `n = 100`

Feature clipping:
- QA820 with `n = 100` for `p01`, `p02`, `p05`
- QA820 with `n = 300` for `p05`

Interpretation:
- baseline is the main result
- feature clipping is exploratory

## 7. How to run

Run one experiment:
```bash
python scripts/run_experiment.py --config configs/baseline_qa820_n300.yaml
```

Run the complete pipeline:
```bash
bash run_all.sh extended
```

Clean outputs:
```bash
bash scripts/clean_outputs.sh --full-repro
```

Build final tables:
```bash
python scripts/build_tables.py
```

Build final figures:
```bash
python scripts/build_figures.py
```

## 8. Final outputs

Main output folders:
- `results/local_runs/` run-level artifacts
- `results/final/tables/` final CSV tables
- `results/final/summaries/` final JSON summaries
- `results/final/figures/` final figures for the report
- `results/final/cases/` selected case analysis outputs

Typical run artifacts:
- `config.json`
- `indices.json`
- `results.json`
- `summary.json`

## 9. Main findings

Main takeaways:
- the INSIDE-EigenScore baseline provides a useful signal for question-level hallucination risk
- performance degrades reasonably when scaling from small subsets to QA820 full scale
- TriviaQA is harder than QA820 for this baseline
- feature clipping has real but mixed effects and does not clearly replace the baseline

Reference summary:
- QA820 baseline remains around AUROC `0.79` at full scale
- TriviaQA baseline is weaker than QA820
- feature clipping changes some cases but does not yield a clear global win

## 10. Limitations

Main limitations:
- requires access to hidden states
- requires multiple generations per question
- feature clipping depends on the calibration protocol
- this repository does not reproduce the full INSIDE paper

## 11. Reproducibility notes

Main settings:
- main model: `meta-llama/Llama-3.2-3B-Instruct`
- main baseline setting: `K = 20`
- feature clipping kept separate from the baseline
- fixed dataset seed and generation seeds through configs

Hardware note:
- the pipeline was run on GPU hardware such as L40S and H100
- runtime depends strongly on hardware and model access

## 12. Project context

This repository is one contribution inside a larger team project on hallucination detection using uncertainty-based methods.

This part focuses specifically on INSIDE-EigenScore and on feature clipping as a targeted exploratory extension.

*Nicolas RINCON*
