"""Call the running Census income API with real HTTP requests.

Usage::

    python request.py --base-url http://127.0.0.1:8000

The script calls ``GET /`` and ``POST /predict`` (once per prediction
class), prints every status code and response body, and exits with status
1 when a response is unexpected or 2 when the server cannot be reached.
"""

import argparse
import json
import sys
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 10.0
PREDICT_PATH = "/predict"

# Copies of starter/examples.py so this script has no project imports.
HIGH_INCOME_EXAMPLE = {
    "age": 44,
    "workclass": "Private",
    "fnlgt": 198282,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}

LOW_INCOME_EXAMPLE = {
    "age": 18,
    "workclass": "Private",
    "fnlgt": 183930,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Other-service",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 12,
    "native-country": "United-States",
}

CASES = (
    ("earns more than 50K", HIGH_INCOME_EXAMPLE, ">50K"),
    ("earns 50K or less", LOW_INCOME_EXAMPLE, "<=50K"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line options."""
    parser = argparse.ArgumentParser(
        description="Send real GET and POST requests to the running API."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"root URL of the running API (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"seconds to wait for each response (default: {DEFAULT_TIMEOUT})",
    )
    return parser.parse_args(argv)


def build_session() -> requests.Session:
    """Create a session that ignores proxy settings from the environment."""
    session = requests.Session()
    session.trust_env = False
    return session


def body_text(response: Any) -> str:
    """Pretty-print a JSON body, falling back to the raw text."""
    try:
        return json.dumps(response.json(), indent=2)
    except ValueError:
        return response.text


def check_root(http: Any, base_url: str, timeout: float) -> bool:
    """GET / and confirm a 200 response carrying a welcome message."""
    url = f"{base_url}/"
    print(f"GET {url}")
    response = http.get(url, timeout=timeout)
    print(f"status: {response.status_code}")
    print(f"body: {body_text(response)}")
    try:
        has_message = "message" in response.json()
    except (ValueError, TypeError):
        has_message = False
    ok = response.status_code == 200 and has_message
    verdict = "OK" if ok else "UNEXPECTED"
    print(f"expected status 200 with a welcome message -> {verdict}")
    return ok


def check_predict(
    http: Any,
    base_url: str,
    payload: dict[str, Any],
    expected: str,
    timeout: float,
) -> bool:
    """POST one record and confirm the expected prediction class."""
    url = f"{base_url}{PREDICT_PATH}"
    print(f"POST {url}")
    print(f"payload: {json.dumps(payload)}")
    response = http.post(url, json=payload, timeout=timeout)
    print(f"status: {response.status_code}")
    print(f"body: {body_text(response)}")
    try:
        prediction = response.json().get("prediction")
    except (ValueError, AttributeError):
        prediction = None
    ok = response.status_code == 200 and prediction == expected
    verdict = "OK" if ok else "MISMATCH"
    print(f"expected prediction: {expected} -> {verdict}")
    return ok


def main(argv: list[str] | None = None, http: Any = None) -> int:
    """Run every check; return 0 on success, 1 on mismatch, 2 if down."""
    args = parse_args(argv)
    base_url = args.base_url.rstrip("/")
    http = http if http is not None else build_session()
    try:
        results = [check_root(http, base_url, args.timeout)]
        for description, payload, expected in CASES:
            print()
            print(f"Case: {description}")
            results.append(
                check_predict(http, base_url, payload, expected, args.timeout)
            )
    except requests.RequestException as error:
        print(f"Could not reach {base_url}: {error}", file=sys.stderr)
        return 2
    print()
    if all(results):
        print("All requests returned the expected results.")
        return 0
    print("Unexpected result from the API; see the output above.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
