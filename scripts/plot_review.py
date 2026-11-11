"""Render saved-result diagnostic figures; never load or execute models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apexpinn.eval.review_figures import build_review_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_review_figures(args.root, args.out)))


if __name__ == "__main__":
    main()
