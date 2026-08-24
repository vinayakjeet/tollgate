"""Fetch the pinned embedding model into models/.

Weights are never committed: they are fetched once, by this script, against an
explicit revision you are forced to name. A floating "latest" would mean the
semantic cache measures one embedding model today and a different one next week,
while every published number quietly claims continuity.

    uv run python scripts/fetch_embedding_model.py \
        --repo sentence-transformers/all-MiniLM-L6-v2 \
        --revision <commit-sha>          # required, on purpose

The second run is a no-op: the snapshot directory is checked for existence
before anything is downloaded, which is also what makes the fetch testable
without network access.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def snapshot_dir(repo: str, revision: str, out_root: Path) -> Path:
    safe_repo = re.sub(r"[^A-Za-z0-9._-]", "__", repo)
    return out_root / f"{safe_repo}__{revision}"


def fetch(repo: str, revision: str, out_root: Path) -> Path:
    target = snapshot_dir(repo, revision, out_root)
    if (target / "config.json").exists():
        print(f"already present, nothing fetched: {target}")
        return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        sys.exit(f"fetching needs huggingface_hub: uv sync --group semantic ({exc})")

    print(f"downloading {repo}@{revision} ...")
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(target),
    )
    print(f"fetched into {path}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision", required=True, help="a commit sha, not 'main'")
    parser.add_argument("--out", default=str(REPO_ROOT / "models"))
    args = parser.parse_args()

    if args.revision in ("main", "master", "latest"):
        sys.exit("refusing a moving revision: pass the full commit sha")
    if len(args.revision) < 12 or not re.fullmatch(r"[0-9a-f]+", args.revision):
        sys.exit("revision must look like a git sha (hex, at least 12 chars)")

    fetch(args.repo, args.revision, Path(args.out))


if __name__ == "__main__":
    main()
