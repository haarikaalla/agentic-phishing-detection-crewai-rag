import json
from pathlib import Path

import joblib
import pytest

from phishguard.cli import main
from phishguard.forensics import forensic_indicators
from phishguard.guidance import retrieve_guidance
from phishguard.model import analyze_email, load_csv, train_and_evaluate

DEMO_CSV = Path(__file__).resolve().parents[1] / "examples" / "demo_emails.csv"


def test_demo_training_writes_reproducible_artifacts(tmp_path):
    rows = load_csv(DEMO_CSV)
    metrics = train_and_evaluate(rows, tmp_path, source="synthetic demo")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert metrics == saved
    assert metrics["training_samples"] + metrics["test_samples"] == 24
    assert 0 <= metrics["average_precision"] <= 1
    assert metrics["dataset_sha256"]
    assert (tmp_path / "phishing_model.joblib").is_file()

    packet = analyze_email(joblib.load(tmp_path / "phishing_model.joblib"), rows[0][0])
    assert 0 <= packet["model_evidence"]["phishing_probability"] <= 1
    assert packet["forensic_indicators"]["url_count"] == 1
    assert packet["retrieved_guidance"]


def test_forensics_never_treats_url_as_action():
    result = forensic_indicators(
        "URGENT! Verify your password at https://bit.ly/example-login now!"
    )
    assert result["shortened_urls"] == ["https://bit.ly/example-login"]
    assert "password" in result["credential_terms"]


def test_guidance_returns_cited_sources():
    results = retrieve_guidance("urgent password verification link")
    assert results
    assert all(item["url"].startswith("https://") for item in results)
    assert all(item["similarity"] > 0 for item in results)


def test_invalid_email_and_labels_are_rejected(tmp_path):
    model_dir = tmp_path / "model"
    train_and_evaluate(load_csv(DEMO_CSV), model_dir, source="synthetic demo")
    model = joblib.load(model_dir / "phishing_model.joblib")
    with pytest.raises(ValueError, match="non-empty"):
        analyze_email(model, " ")
    with pytest.raises(ValueError, match="exceeds"):
        analyze_email(model, "x" * 50_001)

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("text,label\nhello,2\n")
    with pytest.raises(ValueError, match="label"):
        load_csv(bad_csv)


def test_cli_trains_and_analyzes_offline(tmp_path, capsys):
    model_dir = tmp_path / "artifacts"
    assert main(["train", "--csv", str(DEMO_CSV), "--output", str(model_dir)]) == 0
    capsys.readouterr()
    assert main([
        "analyze", "--model", str(model_dir / "phishing_model.joblib"),
        "--text", "URGENT: verify your password immediately",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "model_evidence" in result
    assert "forensic_indicators" in result


def test_conflicting_duplicate_is_excluded(tmp_path):
    rows = load_csv(DEMO_CSV)
    rows.append((rows[0][0].upper(), 0))
    metrics = train_and_evaluate(rows, tmp_path, source="synthetic demo")
    assert metrics["conflicting_duplicates_removed"] == 1
