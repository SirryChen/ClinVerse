#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup(path: Path, ref: str = "29a7d63ce80e9ed2ff83a6060696035538612ccf"):
    path = path.resolve()
    if not path.exists():
        subprocess.run(["git", "clone", "https://github.com/synthetichealth/synthea.git", str(path)], check=True)
        subprocess.run(["git", "checkout", "--detach", ref], cwd=path, check=True)
    if not (path / "gradlew").is_file():
        raise FileNotFoundError("Select a Synthea source checkout")
    subprocess.run(["bash", str(ROOT / "synthea_extensions/deploy.sh"), "--synthea-root", str(path)], check=True)
    subprocess.run(["bash", "gradlew", "uberJar"], cwd=path, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    target = ROOT / "data/preparation/synthea.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"ref": ref, "commit": commit}, indent=2))
    return commit


def main():
    parser = argparse.ArgumentParser(description="Build Synthea with the ClinVerse generation modules")
    parser.add_argument("--synthea-root", type=Path, default=ROOT.parent / "synthea")
    parser.add_argument("--ref", default="29a7d63ce80e9ed2ff83a6060696035538612ccf")
    args = parser.parse_args()
    print(setup(args.synthea_root, args.ref))


if __name__ == "__main__":
    main()
