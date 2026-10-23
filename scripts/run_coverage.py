"""Measure test coverage with coverage.py and save the report as evidence.

Optional local tooling: install it with ``python -m pip install -r
requirements-dev.txt``. The required quality gate (``scripts/check.py``)
does not use coverage, so the Udacity Workspace never needs this script.

Usage::

    python scripts/run_coverage.py [--html] [--output PATH]
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "evidence" / "coverage.txt"
STEPS = (
    ("coverage", "erase"),
    ("coverage", "run", "-m", "pytest", "-q"),
    ("coverage", "report", "-m"),
)


def coverage_available() -> bool:
    """Return True when ``python -m coverage`` works in this interpreter."""
    probe = subprocess.run(
        [sys.executable, "-m", "coverage", "--version"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    return probe.returncode == 0


def run_step(arguments: tuple[str, ...], log: list[str]) -> int:
    """Run ``python -m <arguments>`` from the project root and log it."""
    header = f"$ python -m {' '.join(arguments)}"
    print(header, flush=True)
    result = subprocess.run(
        [sys.executable, "-m", *arguments],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    output = (result.stdout + result.stderr).rstrip()
    if output:
        print(output, flush=True)
    log.append(header)
    if output:
        log.append(output)
    log.append(f"exit status: {result.returncode}")
    log.append("")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    """Run the coverage steps; return the first nonzero exit status."""
    parser = argparse.ArgumentParser(
        description="Run the test suite under coverage.py and save the report."
    )
    parser.add_argument(
        "--html", action="store_true",
        help="also write an HTML report to htmlcov/",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="text file that receives the commands and their output",
    )
    args = parser.parse_args(argv)

    if not coverage_available():
        print(
            "coverage is not installed in this interpreter. Install the "
            "optional tooling with:\n"
            "  python -m pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        return 2

    steps = list(STEPS)
    if args.html:
        steps.append(("coverage", "html"))
    log: list[str] = []
    status = 0
    for step in steps:
        status = run_step(step, log)
        if status:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(log).rstrip() + "\n", encoding="utf-8")
    print(f"Coverage log saved to {args.output}")
    if status:
        print("Coverage run failed; see the output above.", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
