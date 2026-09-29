"""Hugging Face 가중치의 CUDA 양자화, 로컬 저장 및 직접 추론."""

from dataclasses import dataclass
from functools import lru_cache
import math
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
QUANTIZATION_MODES = ("nf4", "int4", "int8", "fp8", "bf16", "fp16", "none")


def normalize_quantization(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"llm.quantization은 {', '.join(QUANTIZATION_MODES)} 중 하나여야 합니다")
    value = value.strip().lower()
    value = {"4bit": "nf4", "8bit": "int8"}.get(value, value)
    if value not in QUANTIZATION_MODES:
        raise ValueError(f"llm.quantization은 {', '.join(QUANTIZATION_MODES)} 중 하나여야 합니다")
    return value


@dataclass(frozen=True)
class ModelSettings:
    source: str
    model: str
    quantization: str
    device: str
    compute_dtype: str
    trust_remote_code: bool


def model_settings(config: dict) -> ModelSettings:
    source = config.get("source", "huggingface")
    if source not in ("huggingface", "local"):
        raise ValueError("llm.source는 huggingface 또는 local이어야 합니다")
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("llm.model에 Hugging Face 모델 ID 또는 로컬 폴더를 지정하세요")
    model = model.strip()
    if source == "local":
        path = Path(model).expanduser()
        model = str((path if path.is_absolute() else ROOT / path).resolve())
    quantization = normalize_quantization(config.get("quantization", "nf4"))
    device = config.get("device", "cuda:0")
    if not isinstance(device, str) or not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("llm.device는 cuda:0 형태여야 합니다")
    dtype = config.get("compute_dtype", "auto")
    if dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("llm.compute_dtype은 auto, float16, bfloat16 중 하나여야 합니다")
    trust = config.get("trust_remote_code", False)
    if not isinstance(trust, bool):
        raise ValueError("llm.trust_remote_code는 불리언이어야 합니다")
    # 로컬 모델의 양자화 설정은 저장된 config.json에서 읽는다.
    return ModelSettings(source, model, quantization if source == "huggingface" else "saved",
                         device, dtype, trust)


def validate_transformers_config(config: dict) -> None:
    model_settings(config)
    for key, default in (("max_new_tokens", 512), ("max_input_tokens", 4096)):
        value = config.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"llm.{key}는 양의 정수여야 합니다")
    temperature = config.get("temperature", 0.0)
    if (type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature < 0):
        raise ValueError("llm.temperature는 0 이상의 유한한 수여야 합니다")
    if not isinstance(config.get("enable_thinking", False), bool):
        raise ValueError("llm.enable_thinking은 불리언이어야 합니다")


@lru_cache(maxsize=1)
def _load_model(settings: ModelSettings):
    """동일 설정의 모델·토크나이저를 프로세스 안에서 재사용한다."""
    if settings.source == "local" and not (Path(settings.model) / "config.json").is_file():
        raise FileNotFoundError(f"로컬 모델 config.json이 없습니다: {settings.model}")
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Transformers 추론 의존성이 필요합니다: uv sync --extra hf") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. NVIDIA 드라이버와 CUDA 지원 PyTorch를 확인하세요")
    device_index = int(settings.device.split(":")[1])
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"존재하지 않는 CUDA 장치입니다: {settings.device}")
    with torch.cuda.device(device_index):
        supports_bf16 = torch.cuda.is_bf16_supported()
    common = {"local_files_only": settings.source == "local",
              "trust_remote_code": settings.trust_remote_code}
    saved_config = AutoConfig.from_pretrained(settings.model, **common)
    quantization = getattr(saved_config, "quantization_config", None)
    if settings.source == "huggingface" and quantization:
        raise ValueError("이미 양자화된 Hub 모델입니다. 직접 양자화하려면 원본 FP16/BF16 모델을 지정하세요")
    mode = settings.quantization
    if mode in ("saved", "none"):
        # 저장된 FP16/BF16/FP32 자료형을 compute_dtype로 덮어쓰지 않는다.
        dtype = "auto"
        dtype_name = str(getattr(saved_config, "dtype", "auto")).removeprefix("torch.")
    else:
        dtype_name = {"bf16": "bfloat16", "fp16": "float16"}.get(mode, settings.compute_dtype)
        if dtype_name == "auto":
            dtype_name = "bfloat16" if supports_bf16 else "float16"
        dtype = getattr(torch, dtype_name)
    if dtype_name == "bfloat16" and not supports_bf16:
        raise ValueError("선택한 GPU가 bfloat16을 지원하지 않습니다. FP16 모델 또는 compute_dtype: float16을 사용하세요")
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    if mode in ("nf4", "int8") or method == "bitsandbytes" or (
        isinstance(quantization, dict) and (quantization.get("load_in_4bit") or quantization.get("load_in_8bit"))
    ):
        try:
            import bitsandbytes  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError("bitsandbytes가 필요합니다: uv sync --extra hf") from exc
    if mode in ("int4", "fp8") or method == "torchao":
        try:
            import torchao  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError("INT4/FP8에는 PyTorch와 호환되는 torchao>=0.15가 필요합니다: uv sync --extra hf") from exc
    tokenizer = AutoTokenizer.from_pretrained(settings.model, **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("토크나이저에 pad_token과 eos_token이 모두 없습니다")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {**common, "device_map": {"": device_index}, "dtype": dtype}
    if settings.source == "huggingface":
        if mode == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        elif mode == "int8":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif mode in ("int4", "fp8"):
            from transformers import TorchAoConfig
            from torchao.quantization import Int4WeightOnlyConfig, Float8WeightOnlyConfig

            recipe = (Int4WeightOnlyConfig(group_size=128) if mode == "int4"
                      else Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn))
            kwargs["quantization_config"] = TorchAoConfig(quant_type=recipe)
    model = AutoModelForCausalLM.from_pretrained(settings.model, **kwargs)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def load_model(config: dict):
    validate_transformers_config(config)
    return _load_model(model_settings(config))


def save_quantized_model(config: dict, output_dir: str) -> Path:
    """7가지 정밀도 중 하나로 준비해 새 폴더에 저장한다. 기존 폴더는 덮어쓰지 않는다."""
    settings = model_settings(config)
    if settings.source != "huggingface":
        raise ValueError("모델 준비·저장은 source: huggingface 설정을 사용하세요")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("저장할 폴더를 --output 또는 llm.save_dir로 지정하세요")
    output = Path(output_dir.replace("{quantization}", settings.quantization)).expanduser()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    if output.exists():
        raise FileExistsError(f"저장 경로가 이미 있습니다. 새 경로를 지정하세요: {output}")
    tokenizer, model = load_model(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        Path(temporary, "preparation.json").write_text(json.dumps({
            "source_model": settings.model, "quantization": settings.quantization,
            "compute_dtype": settings.compute_dtype, "loaded_dtype": str(model.dtype),
            "device": settings.device,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(temporary).rename(output)
    return output


def generate_answer(messages: list[dict], config: dict) -> str:
    tokenizer, model = load_model(config)
    import torch

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", enable_thinking=config.get("enable_thinking", False))
    prompt_length = inputs["input_ids"].shape[1]
    max_new_tokens = config.get("max_new_tokens", 512)
    if prompt_length > config.get("max_input_tokens", 4096):
        raise ValueError("LLM 입력이 max_input_tokens를 초과했습니다. Gold 문맥이나 대화 이력을 확인하세요")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and prompt_length + max_new_tokens > context_limit:
        raise ValueError("입력과 생성 토큰 수가 모델의 컨텍스트 한도를 초과했습니다")
    inputs = inputs.to(model.device)
    temperature = config.get("temperature", 0.0)
    generation = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "pad_token_id": tokenizer.pad_token_id, "use_cache": True}
    if temperature > 0:
        generation["temperature"] = temperature
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation)
    return tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True).strip()
