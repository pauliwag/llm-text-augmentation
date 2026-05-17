#!/usr/bin/env python3
"""
Data Augmentation Methods
==========================
Implements six augmentation strategies for text classification:
  1. No augmentation (baseline)
  2. EDA - POS-aware synonym replacement via WordNet
  3. Back-translation - English -> French -> English via MarianMT (with sampling)
  4. LLM zero-shot generation
  5. LLM paraphrasing
  6. LLM few-shot (label-conditional) generation

All functions accept a list of (text, label) pairs and return
a list of (augmented_text, label) pairs plus generation statistics.
"""

import os
import re
import gc
import time
import random
import nltk
import torch
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk import pos_tag
from nltk.corpus import wordnet
from openai import OpenAI
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================


# LLM provider: "ollama" or "openai"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
# Ollama model name
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:30b-instruct")
# OpenAI model name
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# Ollama base URL
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Max parallel LLM requests
LLM_MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "4"))

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


# ============================================================================
# 1. NO AUGMENTATION (BASELINE)
# ============================================================================


def no_augmentation(texts, labels, n_aug=5, seed=123):
    """Return empty lists (no synthetic data)."""
    return [], [], {"method": "none", "n_requested": 0, "n_generated": 0}


# ============================================================================
# 2. EDA - POS-AWARE SYNONYM REPLACEMENT
# ============================================================================


def _ensure_nltk_data():
    """Download required NLTK data if not present."""
    for resource, path in [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(path, quiet=True)

    # POS tagger name changed across NLTK versions:
    #   >= 3.9.1: "averaged_perceptron_tagger_eng"
    #   <  3.9.1: "averaged_perceptron_tagger"
    for tagger in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"taggers/{tagger}")
            return  # Already installed
        except LookupError:
            pass
    # Neither found - try downloading newest name first, fall back to old
    for tagger in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
        try:
            nltk.download(tagger, quiet=True)
            return
        except Exception:
            continue


def _get_wordnet_pos(treebank_tag):
    """Map Penn Treebank POS tag to WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def _synonym_replace(text, n_replacements=2, rng=None):
    """
    Replace up to n_replacements words with POS-aware WordNet synonyms.
    Only considers synonyms that share the same part of speech as the
    original word, following the POS-aware variant of EDA (Wei & Zou, 2019).
    """
    if rng is None:
        rng = random.Random()

    words = text.split()
    if len(words) < 3:
        return text

    tagged = pos_tag(words)
    candidates = []

    for i, (w, tag) in enumerate(tagged):
        wn_pos = _get_wordnet_pos(tag)
        if wn_pos is None:
            continue
        syns = wordnet.synsets(w.lower(), pos=wn_pos)
        lemmas = set()
        for s in syns:
            for lem in s.lemmas():
                name = lem.name().lower()
                if name != w.lower() and "_" not in name:
                    lemmas.add(name)
        if lemmas:
            candidates.append((i, list(lemmas)))

    if not candidates:
        return text

    n_replace = min(n_replacements, len(candidates))
    chosen = rng.sample(candidates, n_replace)

    new_words = words.copy()
    for idx, lemmas in chosen:
        new_words[idx] = rng.choice(lemmas)

    return " ".join(new_words)


def eda_augmentation(texts, labels, n_aug=5, seed=123):
    """
    EDA POS-aware synonym replacement augmentation (Wei & Zou, 2019).

    Args:
        texts: list of strings
        labels: list of ints
        n_aug: number of augmented copies per original
        seed: random seed

    Returns:
        aug_texts, aug_labels, stats
    """
    _ensure_nltk_data()
    rng = random.Random(seed)

    aug_texts, aug_labels = [], []
    n_requested = len(texts) * n_aug

    for text, label in tqdm(zip(texts, labels), total=len(texts), desc="EDA"):
        for _ in range(n_aug):
            n_replace = rng.randint(1, 3)
            augmented = _synonym_replace(text, n_replacements=n_replace, rng=rng)
            aug_texts.append(augmented)
            aug_labels.append(label)

    stats = {
        "method": "eda",
        "n_requested": n_requested,
        "n_generated": len(aug_texts),
        "n_failed": 0,
    }
    return aug_texts, aug_labels, stats


# ============================================================================
# 3. BACK-TRANSLATION (MarianMT: en -> fr -> en, with sampling)
# ============================================================================


_bt_models = {}


def _get_device():
    """Get the torch device for MarianMT models."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_bt_models():
    """Load MarianMT translation models onto GPU if available (cached after first call)."""
    if "en_fr" not in _bt_models:
        from transformers import MarianMTModel, MarianTokenizer

        device = _get_device()
        print(f"    Loading MarianMT en->fr model (device: {device})...")
        _bt_models["en_fr_tok"] = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-fr"
        )
        _bt_models["en_fr"] = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-fr"
        ).to(device)

        print(f"    Loading MarianMT fr->en model (device: {device})...")
        _bt_models["fr_en_tok"] = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-fr-en"
        )
        _bt_models["fr_en"] = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-fr-en"
        ).to(device)
    return _bt_models


def unload_bt_models():
    """Free MarianMT models from memory (call after back-translation is done)."""
    global _bt_models
    _bt_models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("    MarianMT models unloaded.")


def _translate_batch(
    texts,
    model,
    tokenizer,
    max_length=128,
    batch_size=32,
    do_sample=False,
    temperature=1.0,
    top_k=50,
):
    """
    Translate a batch of texts using MarianMT.
    Automatically places inputs on the same device as the model.
    Supports sampling for diversity when do_sample=True.
    """
    device = next(model.parameters()).device

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_kwargs = {"max_length": max_length}
            if do_sample:
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": temperature,
                        "top_k": top_k,
                    }
                )
            translated = model.generate(**inputs, **gen_kwargs)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)
    return results


def back_translation(texts, labels, n_aug=5, seed=123):
    """
    Back-translation augmentation: English -> French -> English.
    Uses sampling with varying temperature across rounds to produce
    genuinely diverse paraphrases (beam search is deterministic and
    would produce near-identical outputs every round).

    Args:
        texts, labels: input data
        n_aug: augmentations per sample
        seed: random seed

    Returns:
        aug_texts, aug_labels, stats
    """
    models = _load_bt_models()
    aug_texts, aug_labels = [], []

    # Pair each text with its label for safe per-sample tracking
    text_label_pairs = list(zip(texts, labels))

    for aug_round in range(n_aug):
        temp = 0.7 + 0.3 * (aug_round / max(1, n_aug - 1))  # Span 0.7-1.0 evenly
        print(
            f"    Back-translation round {aug_round + 1}/{n_aug} "
            f"(temperature={temp:.2f})..."
        )
        torch.manual_seed(seed + aug_round)

        round_texts = [t for t, _ in text_label_pairs]
        round_labels = [l for _, l in text_label_pairs]

        # en -> fr (with sampling for diversity)
        fr_texts = _translate_batch(
            round_texts,
            models["en_fr"],
            models["en_fr_tok"],
            batch_size=32,
            do_sample=True,
            temperature=temp,
            top_k=50,
        )
        # fr -> en (with sampling for diversity)
        en_texts = _translate_batch(
            fr_texts,
            models["fr_en"],
            models["fr_en_tok"],
            batch_size=32,
            do_sample=True,
            temperature=temp,
            top_k=50,
        )

        aug_texts.extend(en_texts)
        aug_labels.extend(round_labels)

    stats = {
        "method": "backtranslation",
        "n_requested": len(texts) * n_aug,
        "n_generated": len(aug_texts),
        "n_failed": 0,
    }
    return aug_texts, aug_labels, stats


# ============================================================================
# 4-6. LLM-BASED AUGMENTATION
# ============================================================================


def _get_llm_client():
    """Get an OpenAI-compatible client (works with both OpenAI and Ollama)."""
    if LLM_PROVIDER == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY environment variable for OpenAI provider."
            )
        return OpenAI(api_key=api_key), OPENAI_MODEL
    else:
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"), OLLAMA_MODEL


def _llm_generate(
    client, model, prompt, temperature=0.8, max_tokens=200, max_retries=3
):
    """
    Generate text from an LLM with retry logic.

    Uses /no_think system message to suppress Qwen3's internal reasoning
    blocks and save tokens. This is harmless for non-Qwen models (they
    will simply ignore it or treat it as a normal system message).

    Post-processes the output to strip common LLM artifacts: residual
    <think> blocks, leading scaffolding labels ("Title:", "Headline:",
    "Article:", "Output:", "Here's..."), and surrounding quotation marks.
    Returns None if the cleaned output is too short to be usable.

    Returns:
        generated text (str) or None if all retries fail
    """
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()
            # Safety net: strip any residual Qwen3 thinking blocks
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Clean up common LLM artifacts (case-insensitive)
            text = re.sub(
                r"^(Title:|Headline:|Article:|Output:|Here['\"]?s)",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = text.strip().strip('"').strip("'").strip()
            if len(text) > 20:  # Basic quality filter
                return text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"      LLM call failed after {max_retries} retries: {e}")
    return None


def _run_llm_parallel(tasks, desc="LLM generation"):
    """
    Run LLM generation tasks in parallel using a thread pool.

    Args:
        tasks: list of dicts with keys 'client', 'model', 'prompt',
               'temperature', 'label'

    Returns:
        aug_texts, aug_labels, n_failed
    """
    aug_texts, aug_labels = [], []
    n_failed = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
        future_to_label = {}
        for task in tasks:
            future = executor.submit(
                _llm_generate,
                task["client"],
                task["model"],
                task["prompt"],
                task["temperature"],
            )
            future_to_label[future] = task["label"]

        completed = 0
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            result = future.result()
            completed += 1

            if result:
                aug_texts.append(result)
                aug_labels.append(label)
            else:
                n_failed += 1

            if completed % 100 == 0 or completed == total:
                print(f"    {desc}: {completed}/{total} " f"({n_failed} failed so far)")

    return aug_texts, aug_labels, n_failed


def llm_zero_shot(texts, labels, n_aug=5, seed=123):
    """
    LLM zero-shot generation: prompt the model to generate new
    examples for each class label without providing examples from
    the dataset. Tests whether an LLM can generate useful training
    data purely from its knowledge of the topic.

    Note: The input `texts` are not used in prompts (only labels
    determine class names). The total number of generated samples
    is len(texts) * n_aug to match the volume of other methods.
    Deduplication downstream will remove any identical outputs.

    Args:
        texts, labels: input data (labels used for class names only)
        n_aug: augmentations per sample
        seed: random seed

    Returns:
        aug_texts, aug_labels, stats
    """
    client, model = _get_llm_client()
    rng = random.Random(seed)

    tasks = []
    for text, label in zip(texts, labels):
        class_name = LABEL_NAMES[label]
        for _ in range(n_aug):
            prompt = (
                f"Generate a short news article (2-3 sentences) about a "
                f'"{class_name}" topic. The article should be realistic and '
                f"similar in style to news wire services. Output only the "
                f"article text, nothing else."
            )
            tasks.append(
                {
                    "client": client,
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0.7 + rng.random() * 0.3,
                    "label": label,
                }
            )

    aug_texts, aug_labels, n_failed = _run_llm_parallel(tasks, desc="Zero-shot")

    # Check label distribution
    label_dist = Counter(aug_labels)
    print(f"    Zero-shot label distribution: {dict(label_dist)}")
    if n_failed > 0:
        print(
            f"    Warning: {n_failed}/{len(tasks)} generations failed "
            f"({n_failed / len(tasks) * 100:.1f}%)"
        )

    stats = {
        "method": "llm_zero_shot",
        "n_requested": len(tasks),
        "n_generated": len(aug_texts),
        "n_failed": n_failed,
        "label_distribution": dict(label_dist),
    }
    return aug_texts, aug_labels, stats


def llm_paraphrase(texts, labels, n_aug=5, seed=123):
    """
    LLM paraphrasing: prompt the model to rephrase each original text
    while preserving its meaning and class label.

    Note: The prompt includes the class name, which may inject
    class-stereotypical language. This is a realistic augmentation
    setup but worth discussing in analysis.

    Args:
        texts, labels: input data
        n_aug: augmentations per sample
        seed: random seed

    Returns:
        aug_texts, aug_labels, stats
    """
    client, model = _get_llm_client()
    rng = random.Random(seed)

    tasks = []
    for text, label in zip(texts, labels):
        text_snippet = text[:500]
        class_name = LABEL_NAMES[label]

        for _ in range(n_aug):
            prompt = (
                f"Paraphrase the following {class_name} news article. "
                f"Preserve the core meaning but change the wording, "
                f"sentence structure, and style. Output only the "
                f"paraphrased text, nothing else.\n\n"
                f"Original: {text_snippet}"
            )
            tasks.append(
                {
                    "client": client,
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0.6 + rng.random() * 0.4,
                    "label": label,
                }
            )

    aug_texts, aug_labels, n_failed = _run_llm_parallel(tasks, desc="Paraphrase")

    label_dist = Counter(aug_labels)
    print(f"    Paraphrase label distribution: {dict(label_dist)}")
    if n_failed > 0:
        print(
            f"    Warning: {n_failed}/{len(tasks)} generations failed "
            f"({n_failed / len(tasks) * 100:.1f}%)"
        )

    stats = {
        "method": "llm_paraphrase",
        "n_requested": len(tasks),
        "n_generated": len(aug_texts),
        "n_failed": n_failed,
        "label_distribution": dict(label_dist),
    }
    return aug_texts, aug_labels, stats


def llm_few_shot(texts, labels, n_aug=5, seed=123):
    """
    LLM few-shot (label-conditional) generation: provide 2-3 examples
    of each class and ask the model to generate new, diverse examples.

    Args:
        texts, labels: input data
        n_aug: augmentations per sample
        seed: random seed

    Returns:
        aug_texts, aug_labels, stats
    """
    client, model = _get_llm_client()
    rng = random.Random(seed)

    # Group texts by label for example selection
    by_label = {}
    for text, label in zip(texts, labels):
        by_label.setdefault(label, []).append(text)

    tasks = []
    for text, label in zip(texts, labels):
        class_name = LABEL_NAMES[label]
        pool = [t for t in by_label[label] if t != text]
        n_examples = min(2, len(pool))
        examples = rng.sample(pool, n_examples) if pool else []

        examples_str = ""
        for j, ex in enumerate(examples, 1):
            examples_str += f"\nExample {j}: {ex[:300]}"

        for _ in range(n_aug):
            prompt = (
                f'Here are examples of "{class_name}" news articles:'
                f"{examples_str}\n\n"
                f"Example {n_examples + 1}: {text[:300]}\n\n"
                f'Generate a new, different "{class_name}" news article '
                f"in a similar style but with different content. "
                f"Output only the article text, nothing else."
            )
            tasks.append(
                {
                    "client": client,
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0.6 + rng.random() * 0.4,
                    "label": label,
                }
            )

    aug_texts, aug_labels, n_failed = _run_llm_parallel(tasks, desc="Few-shot")

    label_dist = Counter(aug_labels)
    print(f"    Few-shot label distribution: {dict(label_dist)}")
    if n_failed > 0:
        print(
            f"    Warning: {n_failed}/{len(tasks)} generations failed "
            f"({n_failed / len(tasks) * 100:.1f}%)"
        )

    stats = {
        "method": "llm_few_shot",
        "n_requested": len(tasks),
        "n_generated": len(aug_texts),
        "n_failed": n_failed,
        "label_distribution": dict(label_dist),
    }
    return aug_texts, aug_labels, stats


# ============================================================================
# AUGMENTATION REGISTRY
# ============================================================================

# Maps method name -> (function, requires_llm)
AUGMENTATION_METHODS = {
    "none": (no_augmentation, False),
    "eda": (eda_augmentation, False),
    "backtranslation": (back_translation, False),
    "llm_zero_shot": (llm_zero_shot, True),
    "llm_paraphrase": (llm_paraphrase, True),
    "llm_few_shot": (llm_few_shot, True),
}


def get_available_methods(include_llm=True):
    """Return list of method names that can be run."""
    if include_llm:
        return list(AUGMENTATION_METHODS.keys())
    return [name for name, (_, req) in AUGMENTATION_METHODS.items() if not req]


def check_llm_available():
    """Test whether the configured LLM provider is reachable."""
    try:
        client, model = _get_llm_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": "Say OK."},
            ],
            max_tokens=5,
        )
        return True
    except Exception as e:
        print(f"  LLM not available ({LLM_PROVIDER}): {e}")
        return False
