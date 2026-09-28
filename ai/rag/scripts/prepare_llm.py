"""Hugging Face 모델을 선택한 정밀도로 준비하고 모델·토크나이저를 로컬에 저장한다."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ai.rag.config import load_config
from ai.llm.transformers_backend import QUANTIZATION_MODES, save_quantized_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="생략하면 llm.save_dir 사용. 프로젝트 루트 기준")
    parser.add_argument("--quantization", type=str.lower, choices=QUANTIZATION_MODES,
                        help="YAML의 llm.quantization을 이번 실행에서만 덮어씁니다")
    args = parser.parse_args()
    config = load_config(args.config)["llm"]
    if args.quantization is not None:
        config["quantization"] = args.quantization
    if config["provider"] != "transformers":
        parser.error("llm.provider: transformers 설정이 필요합니다")
    output = save_quantized_model(config, args.output or config.get("save_dir"))
    print(f"모델과 토크나이저 저장 완료: {output}")
    print('재사용 설정: llm.source: local, llm.model: "' + output.as_posix() + '"')


if __name__ == "__main__":
    main()
