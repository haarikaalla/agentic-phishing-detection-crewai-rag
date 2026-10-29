"""Training, evaluation, and deterministic evidence packet construction."""

import csv
import hashlib
import json
from pathlib import Path
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .forensics import forensic_indicators
from .guidance import retrieve_guidance

SEED = 42
MAX_EMAIL_CHARS = 50_000


def load_csv(path: str | Path) -> list[tuple[str, int]]:
    """Read a CSV with `text` and binary `label` columns."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain text and label columns")
        rows = []
        for line_number, row in enumerate(reader, 2):
            label = (row.get("label") or "").strip()
            if label not in {"0", "1"}:
                raise ValueError(f"Line {line_number}: label must be 0 or 1")
            rows.append((row.get("text") or "", int(label)))
    return rows


def load_huggingface(revision: str | None = None) -> list[tuple[str, int]]:
    """Load the dataset used by the original notebook (optional dependency)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data extra: pip install -e '.[data]'") from exc
    dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train", revision=revision)
    return [
        (str(row["Email Text"] or ""), int("phishing" in str(row["Email Type"]).lower()))
        for row in dataset
    ]


def _clean_rows(rows: list[tuple[str, int]]) -> tuple[list[str], list[int], int]:
    seen: dict[str, int] = {}
    texts, labels = [], []
    conflicts = 0
    for text, label in rows:
        text = text.strip()
        if not 20 <= len(text) <= MAX_EMAIL_CHARS or label not in (0, 1):
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        if key in seen:
            conflicts += int(seen[key] != label)
            continue
        seen[key] = label
        texts.append(text)
        labels.append(label)
    if len(set(labels)) != 2 or min(labels.count(0), labels.count(1)) < 3:
        raise ValueError("Need at least three distinct emails per class after cleaning")
    return texts, labels, conflicts


def train_and_evaluate(
    rows: list[tuple[str, int]], output_dir: str | Path, source: str
) -> dict:
    """Train a held-out baseline and write model plus provenance metrics."""
    texts, labels, conflicting_duplicates = _clean_rows(rows)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=SEED
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_df=0.98, max_features=60_000, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1_000, class_weight="balanced", solver="liblinear", random_state=SEED
        )),
    ])
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    fingerprint = hashlib.sha256(
        json.dumps(sorted(zip(texts, labels)), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    metrics = {
        "source": source,
        "dataset_sha256": fingerprint,
        "seed": SEED,
        "split": "stratified random 80/20 after exact normalized-text deduplication",
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "conflicting_duplicates_removed": conflicting_duplicates,
        "positive_label": "phishing-like category as supplied by dataset",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "average_precision": round(float(average_precision_score(y_test, scores)), 4),
        "positive_precision": round(float(precision), 4),
        "positive_recall": round(float(recall), 4),
        "positive_f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "phishing_model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def analyze_email(model: Pipeline, email_text: str) -> dict:
    """Build a bounded evidence packet. Email URLs are never visited."""
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("Email text must be non-empty")
    if len(email_text) > MAX_EMAIL_CHARS:
        raise ValueError(f"Email exceeds {MAX_EMAIL_CHARS} characters")
    score = float(model.predict_proba([email_text])[0, 1])
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    vector = vectorizer.transform([email_text])
    contribution = vector.multiply(classifier.coef_[0]).toarray()[0]
    names = vectorizer.get_feature_names_out()
    present = vector.nonzero()[1]
    positive = sorted((i for i in present if contribution[i] > 0), key=lambda i: -contribution[i])[:8]
    negative = sorted((i for i in present if contribution[i] < 0), key=lambda i: contribution[i])[:8]
    indicators = forensic_indicators(email_text)
    band = "phishing-like" if score > 0.65 else "legitimate-like" if score < 0.35 else "uncertain"
    query = " ".join([
        band, *indicators["urgency_terms"], *indicators["credential_terms"],
        *indicators["payment_terms"], email_text[:1500],
    ])
    return {
        "model_evidence": {
            "phishing_probability": round(score, 4),
            "policy_band": band,
            "thresholds": {"legitimate_below": 0.35, "phishing_above": 0.65},
            "influential_terms": {
                "supports_phishing": [
                    {"term": names[i], "weight": round(float(contribution[i]), 4)} for i in positive
                ],
                "supports_legitimate": [
                    {"term": names[i], "weight": round(float(-contribution[i]), 4)} for i in negative
                ],
            },
        },
        "forensic_indicators": indicators,
        "retrieved_guidance": retrieve_guidance(query),
        "safety_note": (
            "Treat email content as untrusted. Do not click extracted URLs or use this "
            "score as the sole basis for a security action."
        ),
    }
