"""`sfc-data`: generate, validate and hash synthetic datasets.

sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite
sfc-data validate --spec configs/data/default.yaml data/synthetic/default.sqlite
sfc-data hash data/synthetic/default.sqlite
sfc-data labels data/synthetic/default.sqlite
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from smart_financial_coach.data.generator.pipeline import generate
from smart_financial_coach.data.generator.spec import load_spec
from smart_financial_coach.data.generator.sqlite_io import read_sqlite
from smart_financial_coach.data.generator.validate import Report, validate
from smart_financial_coach.data.labels import load_truth


def _print_report(report: Report) -> None:
    for key, value in report.stats.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    for error in report.errors:
        print(f"  ERROR {error}", file=sys.stderr)
    print("validation passed" if report.ok else f"validation failed ({len(report.errors)} errors)")


def _generate(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to replace it", file=sys.stderr)
        return 1
    spec = load_spec(args.spec)
    started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"\r  users {done}/{total}", end="" if done < total else "\n", flush=True)

    dataset = generate(spec, progress=progress)
    elapsed = time.perf_counter() - started
    print(f"generated {len(dataset['transactions']):,} transactions in {elapsed:.1f}s")
    if not args.no_validate:
        report = validate(dataset, spec)
        _print_report(report)
        if not report.ok:
            return 1
    dataset.to_sqlite(out, overwrite=args.force)
    print(f"wrote {out}\ncontent hash {dataset.content_hash()}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    report = validate(read_sqlite(args.db), load_spec(args.spec))
    _print_report(report)
    return 0 if report.ok else 1


def _hash(args: argparse.Namespace) -> int:
    print(read_sqlite(args.db).content_hash())
    return 0


def _labels(args: argparse.Namespace) -> int:
    """Label report (FR-2): counts, tiers and ceilings."""
    for key, value in load_truth(args.db).report().items():
        count = key.startswith(("unusual_", "spikes_"))
        print(f"  {key}: {int(value)}" if count else f"  {key}: {value:.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sfc-data", description="Synthetic data generator (FR-1) and labels (FR-2)."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    gen = commands.add_parser(
        "generate", help="generate a dataset from a spec and write it to SQLite"
    )
    gen.add_argument("--spec", type=Path, required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--force", action="store_true", help="replace an existing output file")
    gen.add_argument("--no-validate", action="store_true", help="skip the data quality checks")
    gen.set_defaults(handler=_generate)

    val = commands.add_parser("validate", help="run the data quality checks on a SQLite dataset")
    val.add_argument("--spec", type=Path, required=True)
    val.add_argument("db", type=Path)
    val.set_defaults(handler=_validate)

    hsh = commands.add_parser("hash", help="print a dataset's content hash")
    hsh.add_argument("db", type=Path)
    hsh.set_defaults(handler=_hash)

    lbl = commands.add_parser("labels", help="report label counts, tiers and oracle ceilings")
    lbl.add_argument("db", type=Path)
    lbl.set_defaults(handler=_labels)

    args = parser.parse_args(argv)
    result: int = args.handler(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
