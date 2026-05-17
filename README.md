# LLM-Based Data Augmentation for Low-Resource Text Classification

## Overview

This project empirically evaluates five data augmentation strategies for low-resource text classification on AG News, comparing traditional methods (EDA synonym replacement, back-translation) against LLM-based approaches (zero-shot generation, paraphrasing, few-shot generation). A no-augmentation baseline serves as a reference point.

## Requirements

```bash
pip install -r requirements.txt
```

**For LLM-based methods** (optional - non-LLM methods run without this):

*Option A: Ollama (free, local - used in our experiments)*
```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen3:30b-instruct
# The code connects to http://localhost:11434 by default
```

*Option B: OpenAI API*
```bash
export OPENAI_API_KEY="your-key-here"
export LLM_PROVIDER="openai"
```

## Reproducing Results

```bash
# Run all experiments (detects LLM availability automatically)
python -u run_experiments.py 2>&1 | tee run.log

# Or run without LLM methods (no Ollama/OpenAI needed)
python run_experiments.py --no-llm

# Generate all figures and LaTeX table from results
python generate_plots.py
```

Expected runtime: ~13 hours with GPU and Ollama (LLM generation dominates). Non-LLM methods only: ~40 minutes with GPU.

Reference hardware: NVIDIA RTX 4090 (24 GB VRAM), Ubuntu 24.04.

## File Structure

| File | Description |
|------|-------------|
| `shared_utils.py` | Data loading, DistilBERT training/evaluation, deduplication, label filtering, error analysis |
| `augmentation.py` | All 6 augmentation method implementations (baseline, EDA, back-translation, 3 LLM methods) |
| `run_experiments.py` | Experiment orchestration, augmentation caching, results aggregation |
| `generate_plots.py` | Figure generation and LaTeX table from saved JSON results |
| `requirements.txt` | Python dependencies |

## Output Files

| File | Description |
|------|-------------|
| `results/results.json` | Per-seed results (accuracy, F1, per-class F1, augmentation stats) |
| `results/results_aggregated.json` | Mean ± std across 3 seeds |
| `results/results_with_predictions.json` | Full results including test set predictions (for error analysis) |
| `results/results_table.tex` | LaTeX table for the paper |
| `results/augmentation_samples.json` | Sample augmented texts for qualitative inspection |
| `results/error_analysis.json` | Test examples where augmentation helped or hurt vs. baseline |
| `results/figures/fig1_accuracy_f1_vs_k.png` | Accuracy and F1 vs. training size (two-panel line plot) |
| `results/figures/fig2_accuracy_bars.png` | Grouped bar chart at each *k* |
| `results/figures/fig3_per_class_f1.png` | Per-class F1 heatmap |
| `results/figures/fig4_improvement_over_base.png` | Improvement over baseline (percentage points) |
| `results/figures/fig5_augmentation_volume.png` | Accuracy vs. effective training set size |

## Experimental Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Seeds | 123, 456, 789 | Random seeds for subsampling and model init |
| *k* values | 50, 100, 200 | Labeled examples per class |
| n_aug | 5 | Augmented samples per original |
| Epochs | 5 | DistilBERT fine-tuning epochs |
| Batch size | 16 | Training batch size |
| Learning rate | 2e-5 | Adam optimizer |
| Classifier | distilbert-base-uncased | HuggingFace model |

### LLM Configuration

| Variable | Value Used | Description |
|----------|-----------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend |
| `OLLAMA_MODEL` | `qwen3:30b-instruct` | Model with thinking disabled |
| `LLM_MAX_WORKERS` | `4` | Parallel LLM requests |

## Reproducibility Notes

- **Augmented data is cached** in `./cache/augmented/` keyed by method, *k*, and seed. Delete this directory to regenerate from scratch.
- AG News and MarianMT models are auto-downloaded on first run.
- NLTK data (WordNet, POS tagger) is auto-downloaded on first run.
- Results are reported as mean ± std across 3 seeds. The seeds jointly control data subsampling and model initialization.
- GPU (CUDA) is used automatically if available; the code runs on CPU but substantially slower.