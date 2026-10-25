#!/usr/bin/env python3
"""Run clinical agent evaluation or training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.harness import main

if __name__ == '__main__':
    main()
