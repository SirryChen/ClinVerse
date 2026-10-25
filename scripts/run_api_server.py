#!/usr/bin/env python3
"""Run the ClinicalEnv HTTP API: uvicorn entry point for out-of-process agent
frameworks. Equivalent to `uvicorn env.api:app --host 0.0.0.0 --port 8000`."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the ClinVerse ClinicalEnv API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["CLINVERSE_CONFIG"] = str(Path(args.config).resolve())
    uvicorn.run("env.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
