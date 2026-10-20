"""
Adaptive Privacy-Preserving Cognitive Memory System (DMCLM)
Individual Research Prototype - Main Entry Point
Researcher: Kavishka | SLIIT Student ID: IT23208230
Component: Adaptive Long-Term Memory Consolidation and Resource-Aware Reasoning Framework
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding across Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
SRC_DIR = BACKEND_DIR / "src"
for path in (BACKEND_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.run_dmclm import run_full_dmclm_pipeline


def main():
    limit = 1500
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        limit = int(sys.argv[1])
    run_full_dmclm_pipeline(limit_per_modality=limit)


if __name__ == "__main__":
    main()
