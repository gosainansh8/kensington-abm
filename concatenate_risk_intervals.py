"""
Concatenate per-run risk-interval CSVs from a BehaviorSpace experiment
output folder into one combined CSV, writing the header row only once.

Works for any of the finalize-*-intervals exports (risk-intervals,
otp-enrollment-intervals, ssp-visit-intervals, ...) as long as each
run's file shares the same header and was written with
export-run-risk-intervals-style naming: <prefix>-run-<N>.csv

Usage:
    python concatenate_risk_intervals.py output/base-od
    python concatenate_risk_intervals.py output/base-od \
        --pattern "risk-intervals-run-*.csv" \
        --out output/base-od/risk-intervals-combined.csv

    Or just set DEFAULT_FOLDER below to a hardcoded path and run with no
    arguments at all:
        python concatenate_risk_intervals.py
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# EDIT THIS if you want to hardcode a path instead of passing one on the
# command line. Use the FULL path, e.g.:
#   DEFAULT_FOLDER = "data/base-od"
# Leave as None to require the folder as a command-line argument instead.
DEFAULT_FOLDER = None


def run_number(path: Path) -> int:
    """Extract the run number from a filename like risk-intervals-run-12.csv
    so files are concatenated in run order, not filesystem/glob order."""
    m = re.search(r"-run-(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate per-run risk-interval CSVs, header written once.")
    parser.add_argument("folder", type=Path, nargs="?",
                         default=Path(DEFAULT_FOLDER) if DEFAULT_FOLDER else None,
                         help="Folder containing the per-run CSVs "
                              "(e.g. output/base-od). Optional if "
                              "DEFAULT_FOLDER is set above.")
    parser.add_argument("--pattern", default="risk-intervals-run-*.csv",
                         help="Glob pattern to match per-run files "
                              "(default: risk-intervals-run-*.csv).")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output path (default: combined.csv saved "
                              "next to this script, not inside the input "
                              "folder).")
    args = parser.parse_args()

    if args.folder is None:
        print("ERROR: no folder given, and DEFAULT_FOLDER is not set at "
              "the top of the script. Either pass a folder on the command "
              "line or edit DEFAULT_FOLDER.", file=sys.stderr)
        sys.exit(1)

    if not args.folder.is_dir():
        print(f"ERROR: {args.folder} is not a directory.", file=sys.stderr)
        sys.exit(1)

    files = sorted(args.folder.glob(args.pattern), key=run_number)
    if not files:
        print(f"ERROR: no files matching '{args.pattern}' found in "
              f"{args.folder}.", file=sys.stderr)
        sys.exit(1)

    # Saved next to THIS SCRIPT by default, not inside the input folder --
    # Path(__file__).resolve().parent is the script's own directory
    # regardless of what folder you ran it from or where the input data is.
    script_dir = Path(__file__).resolve().parent
    out_path = args.out or (script_dir / "combined.csv")

    header = None
    n_rows = 0
    n_files = 0
    with out_path.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        for fp in files:
            with fp.open(newline="") as in_f:
                reader = csv.reader(in_f)
                try:
                    file_header = next(reader)
                except StopIteration:
                    print(f"  WARNING: {fp.name} is empty, skipping.")
                    continue

                if header is None:
                    # First file: keep its header.
                    header = file_header
                    writer.writerow(header)
                elif file_header != header:
                    # Header mismatch across files is almost always a sign
                    # you're mixing two different export types (e.g.
                    # risk-intervals and otp-enrollment-intervals) in one
                    # folder. Fail loudly rather than silently misaligning
                    # columns.
                    print(f"ERROR: {fp.name} has a different header than "
                          f"{files[0].name}:\n"
                          f"  expected: {header}\n"
                          f"  found:    {file_header}\n"
                          f"Refusing to concatenate mismatched files.",
                          file=sys.stderr)
                    sys.exit(2)
                # else: matching header, dropped (not written again)

                n_files += 1
                for row in reader:
                    writer.writerow(row)
                    n_rows += 1

    print(f"Concatenated {n_files} files ({n_rows} data rows) into "
          f"{out_path.resolve()}")


if __name__ == "__main__":
    main()