#!/usr/bin/env python3
"""
Shared Utilities
=====================================
Provides:
  - AG News data loading and low-resource subsampling
  - DistilBERT classifier training and evaluation
  - Metric computation (accuracy, macro F1, per-class F1)
  - Label fidelity filtering for augmented data
  - Error analysis utilities

All experiments import from this file to ensure consistency.
"""

import os
import torch
import numpy as np
from datasets import (
    load_dataset,
    Dataset,
    concatenate_datasets,
    ClassLabel,
    Features,
    Value,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import logging
import shutil

# Suppress noisy logs from transformers/datasets
logging.basicConfig(level=logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_CACHE"] = "./cache/datasets"


# ============================================================================
# CONFIGURATION
# ============================================================================


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS = 4
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None


def get_tokenizer():
    """Lazy-load tokenizer (avoids repeated downloads)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


# ============================================================================
# DATA LOADING
# ============================================================================


def load_ag_news():
    """
    Load AG News dataset from HuggingFace.

    Returns:
        train_dataset, test_dataset (HuggingFace Dataset objects)
        Each example has 'text' (str) and 'label' (int 0-3).
    """
    ds = load_dataset("ag_news")
    return ds["train"], ds["test"]


def subsample(dataset, k, seed=123):
    """
    Subsample k examples per class from a HuggingFace Dataset.

    Uses vectorised numpy operations on the label column rather than
    iterating over every example in Python, which is ~100x faster on
    the full AG News training set (120 k rows).

    Args:
        dataset: HuggingFace Dataset with 'text' and 'label' columns
        k: number of examples per class
        seed: random seed for reproducibility

    Returns:
        HuggingFace Dataset with k * NUM_LABELS examples
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset["label"])
    indices = []
    for label in range(NUM_LABELS):
        label_indices = np.where(labels == label)[0]
        chosen = rng.choice(
            label_indices, size=min(k, len(label_indices)), replace=False
        )
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return dataset.select(indices)


def texts_labels_to_dataset(texts, labels):
    """Convert Python lists to a HuggingFace Dataset with ClassLabel feature."""
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=LABEL_NAMES),
        }
    )
    return Dataset.from_dict(
        {"text": list(texts), "label": list(labels)},
        features=features,
    )


def merge_datasets(ds_a, ds_b):
    """Merge two HuggingFace Datasets (concatenate rows)."""
    return concatenate_datasets([ds_a, ds_b])


# ============================================================================
# TOKENIZATION
# ============================================================================


def tokenize_dataset(dataset):
    """
    Tokenize a HuggingFace Dataset for DistilBERT.

    Args:
        dataset: must have 'text' and 'label' columns

    Returns:
        tokenized dataset ready for Trainer
    """
    tokenizer = get_tokenizer()

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenized


# ============================================================================
# CLASSIFIER TRAINING AND EVALUATION
# ============================================================================


def compute_metrics(pred):
    """Compute accuracy and macro F1 for Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def train_and_evaluate(train_dataset, test_dataset, seed, n_epochs=5, batch_size=16):
    """
    Fine-tune a fresh DistilBERT on train_dataset and evaluate on test_dataset.

    Args:
        train_dataset: HuggingFace Dataset with 'text' and 'label'
        test_dataset: HuggingFace Dataset with 'text' and 'label'
        seed: random seed
        n_epochs: training epochs
        batch_size: training batch size

    Returns:
        dict with 'accuracy', 'macro_f1', 'per_class_f1' (list of 4 floats),
        and 'predictions' (array of predicted labels on test set)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Tokenize
    train_tok = tokenize_dataset(train_dataset)
    test_tok = tokenize_dataset(test_dataset)

    # Fresh model each time
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    output_dir = f"./tmp_trainer_{seed}_{os.getpid()}"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=seed,
        report_to="none",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Per-class F1 and predictions
    preds_output = trainer.predict(test_tok)
    pred_labels = preds_output.predictions.argmax(-1)
    true_labels = preds_output.label_ids
    per_class = f1_score(true_labels, pred_labels, average=None).tolist()

    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    return {
        "accuracy": eval_results["eval_accuracy"],
        "macro_f1": eval_results["eval_macro_f1"],
        "per_class_f1": per_class,
        "predictions": pred_labels.tolist(),
        "true_labels": true_labels.tolist(),
    }


# ============================================================================
# LABEL FIDELITY FILTERING
# ============================================================================


def filter_by_label_agreement(aug_texts, aug_labels, real_texts, real_labels):
    """
    Remove augmented samples where a lightweight classifier disagrees
    with the assigned label. Uses TF-IDF + logistic regression trained
    on the real data as a quick sanity check.

    Args:
        aug_texts: list of augmented text strings
        aug_labels: list of corresponding integer labels
        real_texts: list of real training text strings
        real_labels: list of corresponding integer labels

    Returns:
        filtered_texts, filtered_labels, n_removed
    """
    if not aug_texts:
        return [], [], 0

    vec = TfidfVectorizer(max_features=5000)
    X_real = vec.fit_transform(list(real_texts))
    clf = LogisticRegression(max_iter=1000, random_state=123)
    clf.fit(X_real, list(real_labels))

    X_aug = vec.transform(aug_texts)
    preds = clf.predict(X_aug)

    filtered_texts, filtered_labels = [], []
    for text, label, pred in zip(aug_texts, aug_labels, preds):
        if pred == label:
            filtered_texts.append(text)
            filtered_labels.append(label)

    n_removed = len(aug_texts) - len(filtered_texts)
    return filtered_texts, filtered_labels, n_removed


# ============================================================================
# DEDUPLICATION
# ============================================================================


def deduplicate_augmented(aug_texts, aug_labels, real_texts):
    """
    Remove augmented samples that duplicate real data or each other.

    Args:
        aug_texts: list of augmented text strings
        aug_labels: list of corresponding integer labels
        real_texts: list/iterable of real training text strings

    Returns:
        deduped_texts, deduped_labels, n_removed
    """
    real_set = set(t.strip().lower() for t in real_texts)
    seen = set()
    deduped_texts, deduped_labels = [], []

    for t, l in zip(aug_texts, aug_labels):
        t_norm = t.strip().lower()
        if t_norm not in real_set and t_norm not in seen:
            seen.add(t_norm)
            deduped_texts.append(t)
            deduped_labels.append(l)

    n_removed = len(aug_texts) - len(deduped_texts)
    return deduped_texts, deduped_labels, n_removed


# ============================================================================
# ERROR ANALYSIS
# ============================================================================


def error_analysis(
    test_dataset, baseline_preds, augmented_preds, true_labels, n_examples=10
):
    """
    Identify test examples where augmentation helped or hurt compared
    to baseline. Useful for qualitative discussion in the report.

    Args:
        test_dataset: HuggingFace Dataset with 'text' and 'label'
        baseline_preds: list of predicted labels from baseline model
        augmented_preds: list of predicted labels from augmented model
        true_labels: list of true labels
        n_examples: max examples to return per category

    Returns:
        dict with 'helped' and 'hurt' lists, each containing dicts
        with 'text', 'true_label', 'baseline_pred', 'augmented_pred'
    """
    helped, hurt = [], []

    for i in range(len(true_labels)):
        base_correct = baseline_preds[i] == true_labels[i]
        aug_correct = augmented_preds[i] == true_labels[i]

        if aug_correct and not base_correct:
            helped.append(
                {
                    "text": test_dataset[i]["text"][:200],
                    "true_label": LABEL_NAMES[true_labels[i]],
                    "baseline_pred": LABEL_NAMES[baseline_preds[i]],
                    "augmented_pred": LABEL_NAMES[augmented_preds[i]],
                }
            )
        elif base_correct and not aug_correct:
            hurt.append(
                {
                    "text": test_dataset[i]["text"][:200],
                    "true_label": LABEL_NAMES[true_labels[i]],
                    "baseline_pred": LABEL_NAMES[baseline_preds[i]],
                    "augmented_pred": LABEL_NAMES[augmented_preds[i]],
                }
            )

    return {
        "helped": helped[:n_examples],
        "hurt": hurt[:n_examples],
        "n_helped_total": len(helped),
        "n_hurt_total": len(hurt),
    }
