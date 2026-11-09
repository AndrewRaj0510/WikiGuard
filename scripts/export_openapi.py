"""Write the API's OpenAPI schema to frontend/openapi.json, which the frontend's types come from.

The schema depends only on the code, so this builds the app with throwaway settings, reads no .env,
and never connects to a database: create_app opens the pool only when the app starts serving.

    python scripts/export_openapi.py            # rewrite frontend/openapi.json
    python scripts/export_openapi.py --check    # exit 1 if the file is out of date (for CI)

Then, in frontend/:  npm run types
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "openapi.json"
sys.path.insert(0, str(ROOT / "services" / "api"))

from wikiguard_api.app import create_app  # noqa: E402 - needs the path above
from wikiguard_api.foundations import Settings  # noqa: E402

DUMMY = "x" * 48


def schema_text() -> str:
    settings = Settings(  # type: ignore[call-arg] - every required field is given here
        _env_file=None,
        database_url="postgresql://unused@127.0.0.1:1/unused",
        session_secret_key=DUMMY,
        user_password_hash=DUMMY,
        metrics_bearer_token=DUMMY,
    )
    schema = create_app(settings).openapi()
    # Sorted keys and a final newline, so the file only changes when the contract does.
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = parser.parse_args()
    text = schema_text()
    if args.check:
        current = OUT.read_text("utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of date: run scripts/export_openapi.py")  # noqa: T201
            return 1
        return 0
    OUT.write_text(text, "utf-8", newline="\n")
    print(f"Wrote {OUT.relative_to(ROOT)}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
