"""Run the end-to-end Compose stack and leave it serving, for working on the frontend.

The same stack as scripts/compose_e2e.py: real images and least-privilege logins, recorded Wikimedia
fixtures in place of the live stream, and throwaway secrets, so it reads no .env. That script stops
the API as one of its checks and deletes its secrets; this one keeps both, so the API stays up.

    python scripts/fixture_stack.py          # start, wait until ready, print the edit count
    python scripts/fixture_stack.py --down   # stop it and delete its volumes

The API is at http://127.0.0.1:8765, with Explain switched off. Throwaway values, including the
`user` password, go to data/fixture_stack/ (git-ignored), and nowhere else.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_e2e as e2e  # noqa: E402 - shares its Stack and settings

FOLDER = Path("data/fixture_stack")
WAIT_SECONDS = 300


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--down", action="store_true", help="stop the stack and delete its data")
    parser.add_argument("--runs", type=Path, help="the models to serve, in place of data/runs")
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    env_file = FOLDER / "e2e.env"

    if args.down:
        if env_file.exists():
            e2e.Stack(env_file).clean()
        print("Stopped")  # noqa: T201
        return 0

    env_file, user_password = e2e.write_env(FOLDER, args.runs)
    (FOLDER / "user_password.txt").write_text(user_password, "utf-8")
    stack = e2e.Stack(env_file)
    stack.clean()
    stack.run("up", "-d", "--build", timeout=1800)
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{e2e.API_PORT}/ready", timeout=5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(3)
    else:
        print("The API didn't become ready", file=sys.stderr)  # noqa: T201
        return 1
    # The recorded stream takes a little while to be ingested and scored after the API is ready.
    print(f"API ready at http://127.0.0.1:{e2e.API_PORT}; edits are arriving from the fixtures")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
