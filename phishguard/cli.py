"""Reproducible local commands for training and email analysis."""

import argparse
import json
from pathlib import Path

import joblib

from .model import analyze_email, load_csv, load_huggingface, train_and_evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishguard", description="Evidence-first email screening")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="Train and evaluate the text classifier")
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="CSV with text and binary label columns")
    source.add_argument("--huggingface", action="store_true", help="Use the notebook's public dataset")
    train.add_argument("--revision", help="Hugging Face dataset revision for reproducibility")
    train.add_argument("--output", type=Path, default=Path("artifacts"))

    analyze = subcommands.add_parser("analyze", help="Analyze one email without an LLM call")
    analyze.add_argument("--model", type=Path, default=Path("artifacts/phishing_model.joblib"))
    input_group = analyze.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Email text")
    input_group.add_argument("--file", type=Path, help="UTF-8 file containing email text")

    serve = subcommands.add_parser("serve", help="Run a local Gradio demonstration")
    serve.add_argument("--model", type=Path, default=Path("artifacts/phishing_model.joblib"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        if args.revision and not args.huggingface:
            raise SystemExit("--revision is only valid with --huggingface")
        rows = load_huggingface(args.revision) if args.huggingface else load_csv(args.csv)
        source = f"zefang-liu/phishing-email-dataset@{args.revision or 'current'}" if args.huggingface else str(args.csv)
        result = train_and_evaluate(rows, args.output, source)
        print(json.dumps(result, indent=2))
        return 0

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}. Run 'phishguard train' first.")
    # joblib is pickle-based; only load artifacts generated locally from trusted data.
    model = joblib.load(args.model)
    if args.command == "analyze":
        text = args.file.read_text(encoding="utf-8") if args.file else args.text
        print(json.dumps(analyze_email(model, text), indent=2))
        return 0

    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install the UI extra: pip install -e '.[ui]'") from exc
    demo = gr.Interface(
        fn=lambda email: analyze_email(model, email),
        inputs=gr.Textbox(lines=14, label="Email text"),
        outputs=gr.JSON(label="Evidence packet"),
        title="PhishGuard | Evidence-first email screening",
        description="Local demonstration. URLs are treated as text and never opened.",
        flagging_mode="never",
    )
    demo.launch(server_name=args.host, server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
