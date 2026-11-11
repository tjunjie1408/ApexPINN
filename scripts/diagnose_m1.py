"""Inspect saved M1 models on hosted Colab, without fitting or optimizer updates."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, default=Path("data/gold_development_frozen_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--render-only", action="store_true", help="rebuild from saved JSON; no model execution"
    )
    parser.add_argument(
        "--saved", type=Path, help="directory containing all ten per-member diagnostic JSON files"
    )
    parser.add_argument(
        "--process-status", type=Path, help="original process exit receipt to preserve in recovery"
    )
    args = parser.parse_args()
    if args.render_only:
        if args.saved is None:
            parser.error("--render-only requires --saved")
        from apexpinn.eval.offset_diagnostic import recover_diagnostic

        recover_diagnostic(args.saved, args.run, args.out, process_status=args.process_status)
        return 0
    import jax

    jax.config.update("jax_enable_x64", True)
    from apexpinn.eval.offset_diagnostic import run_diagnostic

    run_diagnostic(args.run, args.gold_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
