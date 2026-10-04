"""Pick a healthy OpenRouter key before launching agents (free: no completion is sent).

Each team member has an OpenRouter key. Omnigent providers read a fixed env var, so
the Makefile evals this script to point those vars at a key that is valid and still
has credits:

  OPENROUTER_API_KEY            <- first healthy of OPENROUTER_API_KEY, OPENROUTER_API_KEY_BACKUP
  OPENROUTER_API_KEY_CAMPAIGNS  <- first healthy of OPENROUTER_API_KEY_CAMPAIGNS, OPENROUTER_API_KEY_BACKUP

    python scripts/openrouter_keys.py            # status table (no keys printed)
    eval "$(python scripts/openrouter_keys.py --export)"
"""
import os
import shlex
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

MIN_CREDITS = 0.25  # USD; below this a key is treated as exhausted
ROLES = {
    "OPENROUTER_API_KEY": ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_BACKUP"),
    # Campaigns run only on the friend's keys: never fall back to the demo key (its owner's credits).
    "OPENROUTER_API_KEY_CAMPAIGNS": ("OPENROUTER_API_KEY_CAMPAIGNS", "OPENROUTER_API_KEY_BACKUP"),
}


def health(key):
    """(ok, remaining USD or None, reason)."""
    h = {"Authorization": f"Bearer {key}"}
    try:
        k = requests.get("https://openrouter.ai/api/v1/key", headers=h, timeout=15)
        if k.status_code != 200:
            return False, None, f"key rejected ({k.status_code})"
        info = k.json()["data"]
        remaining = []
        if info.get("limit_remaining") is not None:
            remaining.append(info["limit_remaining"])
        c = requests.get("https://openrouter.ai/api/v1/credits", headers=h, timeout=15)
        if c.status_code == 200:
            d = c.json()["data"]
            remaining.append(d["total_credits"] - d["total_usage"])
    except requests.RequestException as e:
        return False, None, f"unreachable ({e.__class__.__name__})"
    left = min(remaining) if remaining else None
    if left is not None and left < MIN_CREDITS:
        return False, left, "out of credits"
    return True, left, "ok"


def main():
    original = {v: os.getenv(v) for roles in ROLES.values() for v in roles}
    checked = {v: health(k) for v, k in original.items() if k}
    for var, (ok, left, why) in checked.items():
        money = f"${left:.2f} left" if left is not None else "no limit info"
        print(f"{var:30s} {'OK ' if ok else 'BAD'}  {money:16s} {why}", file=sys.stderr)
    exports = {}
    for target, candidates in ROLES.items():
        pick = next((v for v in candidates if v in checked and checked[v][0]), None)
        if pick is None:
            print(f"!! no healthy OpenRouter key for {target}", file=sys.stderr)
            continue
        if pick != target:
            print(f"-> {target} uses {pick}", file=sys.stderr)
        exports[target] = original[pick]
    if "--export" in sys.argv:
        for var, key in exports.items():
            print(f"export {var}={shlex.quote(key)}")


if __name__ == "__main__":
    main()
