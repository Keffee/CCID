"""Evaluate completed predictions without involving the inference process."""
import argparse
import json
from pathlib import Path
from .io import load_labels
from .metrics import evaluate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("choose a new output path")
    rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines()
            if line.strip()]
    metrics = evaluate_records(rows, load_labels(args.labels), k=args.k)
    destination.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
