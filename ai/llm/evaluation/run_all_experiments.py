"""명시된 모든 실험을 순서대로 실행한다."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ai.llm.evaluation.run_experiment import run
from ai.rag.config import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    for path in sorted((ROOT / "ai/llm/configs/experiments").glob("*.yaml")):
        print(run(str(path), args.sample, args.rebuild))


if __name__ == "__main__":
    main()
