#!/usr/bin/env python3
"""
Main Experiment Runner
=======================
Runs the full data augmentation comparison experiment:
  - For each seed in {123, 456, 789}:
    - For each k in {50, 100, 200}:
      - Subsample k examples per class from AG News training set
      - For each augmentation method:
        - Generate augmented data (with dedup + optional label filtering)
        - Train DistilBERT on real + augmented data
        - Evaluate on full AG News test set
  - Saves all results as JSON
  - Prints summary tables
  - Runs error analysis for qualitative discussion

Usage:
    python run_experiments.py [--no-llm] [--n-aug 5] [--methods none,eda,llm_paraphrase]
                              [--filter-labels] [--skip-samples] [--skip-error-analysis]

Requirements: See requirements.txt
Expected runtime: ~13 hours with GPU, longer on CPU-only
  (LLM augmentation time depends on provider speed)
Reference hardware: NVIDIA RTX 4090 (24 GB VRAM), Ubuntu 24.04
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import gc

from shared_utils import (
    load_ag_news,
    subsample,
    texts_labels_to_dataset,
    merge_datasets,
    train_and_evaluate,
    filter_by_label_agreement,
    deduplicate_augmented,
    error_analysis,
    DEVICE,
    LABEL_NAMES,
)
from augmentation import (
    AUGMENTATION_METHODS,
    get_available_methods,
    check_llm_available,
    unload_bt_models,
    eda_augmentation,
    back_translation,
    llm_paraphrase,
    llm_few_shot,
)

# ============================================================================
# CONFIGURATION
# ============================================================================


SEEDS = [123, 456, 789]
K_VALUES = [50, 100, 200]
N_AUG = 5  # Augmented samples per original
N_EPOCHS = 5  # DistilBERT fine-tuning epochs
BATCH_SIZE = 16  # Training batch size
RESULTS_FILE = "results.json"
AUG_CACHE_DIR = "./cache/augmented"
OUTPUT_DIR = "results"
FIGURES_DIR = "results/figures"


# ============================================================================
# AUGMENTATION CACHING
# ============================================================================


def _cache_path(method, k, seed):
    """Path for cached augmented data."""
    os.makedirs(AUG_CACHE_DIR, exist_ok=True)
    return os.path.join(AUG_CACHE_DIR, f"{method}_k{k}_seed{seed}.json")


def _load_cached(method, k, seed):
    """Load cached augmented data if available."""
    path = _cache_path(method, k, seed)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data["texts"], data["labels"], data.get("stats", {})
    return None, None, None


def _save_cache(method, k, seed, texts, labels, stats):
    """Cache augmented data to disk."""
    path = _cache_path(method, k, seed)
    with open(path, "w") as f:
        json.dump({"texts": texts, "labels": labels, "stats": stats}, f)


# ============================================================================
# EXPERIMENT
# ============================================================================


def run_single_experiment(
    method_name, aug_fn, train_sub, test_data, k, seed, n_aug, filter_labels=False
):
    """
    Run one augmentation + classification experiment.

    Returns:
        dict with accuracy, macro_f1, per_class_f1, n_train, n_aug_generated,
        augmentation_stats, predictions, true_labels
    """
    real_texts = list(train_sub["text"])
    real_labels = list(train_sub["label"])

    # Check cache
    aug_texts, aug_labels, cached_stats = _load_cached(method_name, k, seed)

    if aug_texts is None:
        # Generate augmented data
        aug_texts, aug_labels, stats = aug_fn(
            real_texts, real_labels, n_aug=n_aug, seed=seed
        )
        # Cache raw output for reuse
        _save_cache(method_name, k, seed, aug_texts, aug_labels, stats)
    else:
        stats = cached_stats or {}
        print(f"  Loaded {len(aug_texts)} cached augmented samples")

    # Post-processing: deduplication
    n_before_dedup = len(aug_texts)
    if aug_texts:
        aug_texts, aug_labels, n_dupes = deduplicate_augmented(
            aug_texts, aug_labels, real_texts
        )
        if n_dupes > 0:
            print(f"  Dedup: removed {n_dupes} duplicates ({n_dupes}/{n_before_dedup})")
        stats["n_duplicates_removed"] = n_dupes

    # Post-processing: optional label fidelity filtering
    if filter_labels and aug_texts:
        n_before_filter = len(aug_texts)
        aug_texts, aug_labels, n_filtered = filter_by_label_agreement(
            aug_texts, aug_labels, real_texts, real_labels
        )
        if n_filtered > 0:
            print(
                f"  Label filter: removed {n_filtered} off-label samples "
                f"({n_filtered}/{n_before_filter} = "
                f"{n_filtered / n_before_filter * 100:.1f}%)"
            )
        stats["n_label_filtered"] = n_filtered

    # Combine real + augmented
    if aug_texts:
        aug_dataset = texts_labels_to_dataset(aug_texts, aug_labels)
        combined = merge_datasets(train_sub, aug_dataset)
    else:
        combined = train_sub

    n_total = len(combined)
    n_generated = len(aug_texts) if aug_texts else 0

    # Train and evaluate
    metrics = train_and_evaluate(
        combined, test_data, seed=seed, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE
    )
    metrics["n_train"] = n_total
    metrics["n_aug_generated"] = n_generated
    metrics["augmentation_stats"] = stats

    return metrics


def run_all_experiments(methods_to_run, n_aug=N_AUG, filter_labels=False):
    """
    Run the full experiment matrix.

    Returns:
        results: nested dict [method][k][seed] = metrics
        test_data: the test dataset (for error analysis)
    """
    print("=" * 70)
    print("LLM Data Augmentation Experiment")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"k values: {K_VALUES}")
    print(f"Augmentations per sample: {n_aug}")
    print(f"Methods: {methods_to_run}")
    print(f"Label filtering: {'ON' if filter_labels else 'OFF'}")
    print(f"Classifier: DistilBERT ({N_EPOCHS} epochs, batch {BATCH_SIZE})")
    print("=" * 70)

    # Load data
    print("\nLoading AG News dataset...")
    train_full, test_data = load_ag_news()
    print(f"  Train: {len(train_full)} examples")
    print(f"  Test:  {len(test_data)} examples")

    results = {}
    total_runs = len(methods_to_run) * len(K_VALUES) * len(SEEDS)
    run_count = 0
    start_time = time.time()

    for method_name in methods_to_run:
        aug_fn, _ = AUGMENTATION_METHODS[method_name]
        results[method_name] = {}

        for k in K_VALUES:
            results[method_name][str(k)] = {}

            for seed in SEEDS:
                run_count += 1
                elapsed = time.time() - start_time
                print(
                    f"\n[{run_count}/{total_runs}] "
                    f"Method={method_name}, k={k}, seed={seed} "
                    f"(elapsed: {elapsed / 60:.1f} min)"
                )

                # Subsample
                train_sub = subsample(train_full, k, seed=seed)
                print(f"  Subsampled {len(train_sub)} examples ({k} per class)")

                # Run experiment
                metrics = run_single_experiment(
                    method_name,
                    aug_fn,
                    train_sub,
                    test_data,
                    k,
                    seed,
                    n_aug,
                    filter_labels=filter_labels,
                )

                results[method_name][str(k)][str(seed)] = metrics
                print(
                    f"  -> Acc: {metrics['accuracy']:.4f} | "
                    f"F1: {metrics['macro_f1']:.4f} | "
                    f"Train size: {metrics['n_train']} "
                    f"({metrics['n_aug_generated']} augmented)"
                )

                gc.collect()

        # Free MarianMT models after all backtranslation runs are done
        if method_name == "backtranslation":
            unload_bt_models()

    total_time = time.time() - start_time
    print(f"\n\nTotal time: {total_time / 60:.1f} minutes")

    return results, test_data, train_full


# ============================================================================
# ERROR ANALYSIS
# ============================================================================


def run_error_analysis(results, test_data):
    """
    For each augmentation method, compare against the baseline to find
    examples where augmentation helped or hurt.

    Prefers k=100, seed=123 but falls back to whatever is available.
    Saves results to error_analysis.json for use in report.
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    if "none" not in results:
        print("No baseline found; skipping error analysis.")
        return

    # Try k=100 / seed=123, fall back to whatever is available
    k_str, seed_str = "100", "123"
    if k_str not in results["none"]:
        k_str = next(iter(results["none"]), None)
    if k_str and seed_str not in results["none"].get(k_str, {}):
        seed_str = next(iter(results["none"].get(k_str, {})), None)
    if not k_str or not seed_str:
        print("Baseline results not found; skipping error analysis.")
        return

    baseline = results["none"][k_str][seed_str]
    if "predictions" not in baseline:
        print("Baseline predictions not found; skipping.")
        return

    print(f"  Using k={k_str}, seed={seed_str} for error analysis")

    true_labels = baseline["true_labels"]
    baseline_preds = baseline["predictions"]
    analysis = {}

    for method in results:
        if method == "none":
            continue
        aug_result = results[method].get(k_str, {}).get(seed_str)
        if not aug_result or "predictions" not in aug_result:
            continue

        ea = error_analysis(
            test_data,
            baseline_preds,
            aug_result["predictions"],
            true_labels,
            n_examples=5,
        )
        analysis[method] = ea
        print(
            f"\n  {method}: +{ea['n_helped_total']} helped, "
            f"-{ea['n_hurt_total']} hurt (net: "
            f"{ea['n_helped_total'] - ea['n_hurt_total']:+d})"
        )

        if ea["helped"]:
            print("    Example helped:")
            ex = ea["helped"][0]
            print(f"      Text: {ex['text'][:120]}...")
            print(
                f"      True: {ex['true_label']} | "
                f"Base: {ex['baseline_pred']} | "
                f"Aug: {ex['augmented_pred']}"
            )

    with open(os.path.join(OUTPUT_DIR, "error_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'error_analysis.json')}")


# ============================================================================
# RESULTS AGGREGATION
# ============================================================================


def aggregate_results(results):
    """
    Compute mean ± std across seeds for each (method, k).

    Returns:
        agg: dict [method][k] = {metric: (mean, std)}
    """
    agg = {}
    for method in results:
        agg[method] = {}
        for k in results[method]:
            seed_metrics = results[method][k]
            accs = [seed_metrics[s]["accuracy"] for s in seed_metrics]
            f1s = [seed_metrics[s]["macro_f1"] for s in seed_metrics]
            pcf1s = np.array([seed_metrics[s]["per_class_f1"] for s in seed_metrics])

            agg[method][k] = {
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "macro_f1_mean": np.mean(f1s),
                "macro_f1_std": np.std(f1s),
                "per_class_f1_mean": np.mean(pcf1s, axis=0).tolist(),
                "per_class_f1_std": np.std(pcf1s, axis=0).tolist(),
                "n_train": list(seed_metrics.values())[0]["n_train"],
            }

    return agg


def print_summary(agg):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY (mean ± std across 3 seeds)")
    print("=" * 90)

    for k in K_VALUES:
        k_str = str(k)
        print(f"\n  k = {k} examples per class:")
        print(
            f"  {'Method':<28} {'Accuracy':>16} {'Macro F1':>16} " f"{'Train Size':>12}"
        )
        print("  " + "-" * 76)

        for method in agg:
            if k_str in agg[method]:
                m = agg[method][k_str]
                print(
                    f"  {method:<28} "
                    f"{m['accuracy_mean']:.4f} ± {m['accuracy_std']:.4f}  "
                    f"{m['macro_f1_mean']:.4f} ± {m['macro_f1_std']:.4f}  "
                    f"{m['n_train']:>10}"
                )

    print("=" * 90)

    # Best method per k
    print("\nBest method per k (by accuracy):")
    for k in K_VALUES:
        k_str = str(k)
        best_method = max(
            agg.keys(),
            key=lambda m: agg[m].get(k_str, {}).get("accuracy_mean", 0),
        )
        best_acc = agg[best_method][k_str]["accuracy_mean"]
        print(f"  k={k}: {best_method} (accuracy = {best_acc:.4f})")


# ============================================================================
# AUGMENTATION SAMPLES FOR REPORT
# ============================================================================


def save_augmentation_samples(train_full, n_display=5):
    """
    Generate and save sample augmentations for qualitative analysis.
    Includes at least one example per class.
    """
    print("\nGenerating sample augmentations for report...")

    # Get examples covering all 4 classes
    train_sub = subsample(train_full, k=5, seed=123)
    # Take first n_display, ensuring class diversity
    seen_labels = set()
    selected_idx = []
    for i in range(len(train_sub)):
        label = train_sub[i]["label"]
        if label not in seen_labels or len(selected_idx) < n_display:
            selected_idx.append(i)
            seen_labels.add(label)
        if len(selected_idx) >= n_display:
            break

    texts = [train_sub[i]["text"] for i in selected_idx]
    labels = [train_sub[i]["label"] for i in selected_idx]

    samples = {
        "originals": [
            {"text": t, "label": LABEL_NAMES[l]} for t, l in zip(texts, labels)
        ]
    }

    # EDA
    eda_t, eda_l, _ = eda_augmentation(texts, labels, n_aug=1, seed=123)
    samples["eda"] = [
        {"text": t, "label": LABEL_NAMES[l]} for t, l in zip(eda_t, eda_l)
    ]

    # Back-translation
    try:
        bt_t, bt_l, _ = back_translation(texts, labels, n_aug=1, seed=123)
        samples["backtranslation"] = [
            {"text": t, "label": LABEL_NAMES[l]} for t, l in zip(bt_t, bt_l)
        ]
        # Free translation models after generating samples
        unload_bt_models()
    except Exception as e:
        print(f"  Back-translation sample failed: {e}")

    # LLM methods
    try:
        if check_llm_available():
            lp_t, lp_l, _ = llm_paraphrase(texts, labels, n_aug=1, seed=123)
            samples["llm_paraphrase"] = [
                {"text": t, "label": LABEL_NAMES[l]} for t, l in zip(lp_t, lp_l)
            ]

            lf_t, lf_l, _ = llm_few_shot(texts, labels, n_aug=1, seed=123)
            samples["llm_few_shot"] = [
                {"text": t, "label": LABEL_NAMES[l]} for t, l in zip(lf_t, lf_l)
            ]
    except Exception as e:
        print(f"  LLM sample generation failed: {e}")

    with open(os.path.join(OUTPUT_DIR, "augmentation_samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'augmentation_samples.json')}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM data augmentation experiments"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-based methods (run only baseline, EDA, back-translation)",
    )
    parser.add_argument(
        "--n-aug",
        type=int,
        default=N_AUG,
        help=f"Augmented samples per original (default: {N_AUG})",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to run (default: all available)",
    )
    parser.add_argument(
        "--filter-labels",
        action="store_true",
        help="Filter augmented samples by label agreement with a TF-IDF classifier",
    )
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Skip generating augmentation samples for report",
    )
    parser.add_argument(
        "--skip-error-analysis",
        action="store_true",
        help="Skip error analysis step",
    )
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Determine which methods to run
    if args.methods:
        methods = args.methods.split(",")
        # Validate method names
        for m in methods:
            if m not in AUGMENTATION_METHODS:
                print(
                    f"ERROR: Unknown method '{m}'. "
                    f"Available: {list(AUGMENTATION_METHODS.keys())}"
                )
                sys.exit(1)
    elif args.no_llm:
        methods = get_available_methods(include_llm=False)
    else:
        llm_ok = check_llm_available()
        methods = get_available_methods(include_llm=llm_ok)
        if not llm_ok:
            print("  -> Running without LLM methods.")

    print(f"\nMethods to run: {methods}")

    # Run experiments
    results, test_data, train_full = run_all_experiments(
        methods, n_aug=args.n_aug, filter_labels=args.filter_labels
    )

    # Save raw results (strip predictions to save space in main file)
    results_for_save = {}
    for method in results:
        results_for_save[method] = {}
        for k in results[method]:
            results_for_save[method][k] = {}
            for s in results[method][k]:
                entry = dict(results[method][k][s])
                # Keep predictions in a separate file for error analysis
                entry.pop("predictions", None)
                entry.pop("true_labels", None)
                results_for_save[method][k][s] = entry

    with open(os.path.join(OUTPUT_DIR, RESULTS_FILE), "w") as f:
        json.dump(results_for_save, f, indent=2)
    print(f"\nSaved raw results: {os.path.join(OUTPUT_DIR, RESULTS_FILE)}")

    # Save full results with predictions for error analysis
    with open(os.path.join(OUTPUT_DIR, "results_with_predictions.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"Saved results with predictions: {os.path.join(OUTPUT_DIR, 'results_with_predictions.json')}"
    )

    # Aggregate and print
    agg = aggregate_results(results)
    print_summary(agg)

    with open(os.path.join(OUTPUT_DIR, "results_aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(
        f"Saved aggregated results: {os.path.join(OUTPUT_DIR, 'results_aggregated.json')}"
    )

    # Error analysis
    if not args.skip_error_analysis and "none" in results:
        run_error_analysis(results, test_data)

    # Generate augmentation samples
    if not args.skip_samples:
        save_augmentation_samples(train_full)

    print("\nDone! Run `python generate_plots.py` to create figures.")


if __name__ == "__main__":
    main()
