"""GPU·다운로드 없이 양자화 저장, 오프라인 재로딩, 생성 호출 계약을 검사한다."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from ai.llm import transformers_backend as backend


class TransformersBackendTests(unittest.TestCase):
    def setUp(self):
        backend._load_model.cache_clear()
        self.config = {"provider": "transformers", "source": "huggingface",
                       "model": "Qwen/Qwen3-1.7B", "quantization": "4bit",
                       "device": "cuda:0", "temperature": 0.0}
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.device_count.return_value = 1
        self.torch.cuda.is_bf16_supported.return_value = True
        self.torch.cuda.device.side_effect = lambda *_: nullcontext()
        self.torch.inference_mode.side_effect = nullcontext
        self.hf = MagicMock()
        self.ao = MagicMock()
        self.hf.AutoConfig.from_pretrained.return_value = SimpleNamespace(quantization_config=None)
        self.model = self.hf.AutoModelForCausalLM.from_pretrained.return_value
        self.model.config = SimpleNamespace(use_cache=False, max_position_embeddings=8192)
        self.model.device = "cuda:0"
        self.tokenizer = self.hf.AutoTokenizer.from_pretrained.return_value
        self.tokenizer.pad_token_id = 2
        self.tokenizer.eos_token_id = 2
        self.tokenizer.eos_token = "<eos>"
        self.patch = patch.dict("sys.modules", {"torch": self.torch, "transformers": self.hf,
                                                "accelerate": MagicMock(), "bitsandbytes": MagicMock(),
                                                "torchao": MagicMock(), "torchao.quantization": self.ao})
        self.patch.start()

    def tearDown(self):
        backend._load_model.cache_clear()
        self.patch.stop()

    def test_hub_quantization_and_cache(self):
        first = backend.load_model(self.config)
        self.assertEqual(backend.load_model(self.config), first)
        self.hf.AutoModelForCausalLM.from_pretrained.assert_called_once()
        self.hf.BitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs = self.hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], {"": 0})
        self.assertFalse(kwargs["local_files_only"])
        self.model.eval.assert_called_once()
        self.assertTrue(self.model.config.use_cache)

    def test_eight_bit_and_unquantized(self):
        for mode in ("8bit", "none"):
            with self.subTest(mode=mode):
                backend._load_model.cache_clear()
                self.hf.BitsAndBytesConfig.reset_mock()
                backend.load_model({**self.config, "quantization": mode})
                kwargs = self.hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                if mode == "8bit":
                    self.hf.BitsAndBytesConfig.assert_called_once_with(load_in_8bit=True)
                else:
                    self.assertNotIn("quantization_config", kwargs)
                    self.hf.BitsAndBytesConfig.assert_not_called()

    def test_all_seven_modes_have_distinct_loading_settings(self):
        expected_dtypes = {"nf4": self.torch.bfloat16, "int4": self.torch.bfloat16,
                           "int8": self.torch.bfloat16, "fp8": self.torch.bfloat16,
                           "bf16": self.torch.bfloat16, "fp16": self.torch.float16, "none": "auto"}
        self.assertEqual(set(expected_dtypes), set(backend.QUANTIZATION_MODES))
        for mode, dtype in expected_dtypes.items():
            with self.subTest(mode=mode):
                backend._load_model.cache_clear()
                self.hf.BitsAndBytesConfig.reset_mock()
                self.hf.TorchAoConfig.reset_mock()
                self.ao.reset_mock()
                backend.load_model({**self.config, "quantization": mode.upper()})
                kwargs = self.hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                self.assertEqual(kwargs["dtype"], dtype)
                if mode == "int4":
                    self.ao.Int4WeightOnlyConfig.assert_called_once_with(group_size=128)
                    self.hf.TorchAoConfig.assert_called_once_with(quant_type=self.ao.Int4WeightOnlyConfig.return_value)
                    self.hf.BitsAndBytesConfig.assert_not_called()
                elif mode == "fp8":
                    self.ao.Float8WeightOnlyConfig.assert_called_once_with(weight_dtype=self.torch.float8_e4m3fn)
                    self.hf.TorchAoConfig.assert_called_once_with(quant_type=self.ao.Float8WeightOnlyConfig.return_value)
                    self.hf.BitsAndBytesConfig.assert_not_called()
                elif mode in ("bf16", "fp16", "none"):
                    self.assertNotIn("quantization_config", kwargs)
                    self.hf.BitsAndBytesConfig.assert_not_called()
                    self.hf.TorchAoConfig.assert_not_called()

    def test_legacy_names_remain_compatible(self):
        self.assertEqual(backend.normalize_quantization("4bit"), "nf4")
        self.assertEqual(backend.normalize_quantization("8bit"), "int8")

    def test_storage_and_offline_reload_for_every_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in backend.QUANTIZATION_MODES:
                with self.subTest(mode=mode):
                    backend._load_model.cache_clear()
                    self.hf.AutoConfig.from_pretrained.return_value = SimpleNamespace(quantization_config=None)
                    saved_quantization = ({"quant_method": "torchao"} if mode in ("int4", "fp8")
                                          else {"quant_method": "bitsandbytes"} if mode in ("nf4", "int8") else None)
                    saved_dtype = "float16" if mode == "fp16" else "bfloat16"

                    def save_model(path, **kwargs):
                        Path(path, "config.json").write_text(json.dumps({
                            "quantization_config": saved_quantization, "dtype": saved_dtype}))

                    self.model.save_pretrained.side_effect = save_model
                    output = backend.save_quantized_model(
                        {**self.config, "quantization": mode}, str(Path(directory) / "model-{quantization}"))
                    self.assertEqual(output.name, f"model-{mode}")
                    metadata = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["quantization"], mode)
                    self.hf.AutoConfig.from_pretrained.return_value = SimpleNamespace(
                        quantization_config=saved_quantization, dtype=saved_dtype)
                    self.hf.TorchAoConfig.reset_mock()
                    self.hf.BitsAndBytesConfig.reset_mock()
                    backend.load_model({**self.config, "source": "local", "model": str(output),
                                        "compute_dtype": "float16", "quantization": "none"})
                    kwargs = self.hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                    self.assertEqual(kwargs["dtype"], "auto")
                    self.assertTrue(kwargs["local_files_only"])
                    self.assertNotIn("quantization_config", kwargs)
                    self.hf.TorchAoConfig.assert_not_called()
                    self.hf.BitsAndBytesConfig.assert_not_called()

    def test_explicit_precision_overrides_compute_dtype(self):
        backend.load_model({**self.config, "quantization": "fp16", "compute_dtype": "bfloat16"})
        self.assertEqual(self.hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs["dtype"], self.torch.float16)
        self.torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            backend.load_model({**self.config, "quantization": "bf16", "compute_dtype": "float16"})

    def test_save_and_local_reload_use_saved_quantization_offline(self):
        def save_model(path, **kwargs):
            Path(path, "config.json").write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}))
            Path(path, "model.safetensors").write_bytes(b"test fixture")

        def save_tokenizer(path):
            Path(path, "tokenizer_config.json").write_text("{}")

        self.model.save_pretrained.side_effect = save_model
        self.tokenizer.save_pretrained.side_effect = save_tokenizer
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "saved"
            self.assertEqual(backend.save_quantized_model(self.config, str(output)), output.resolve())
            self.assertTrue((output / "model.safetensors").is_file())
            self.assertTrue((output / "tokenizer_config.json").is_file())
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            with self.assertRaises(FileExistsError):
                backend.save_quantized_model(self.config, str(output))
            self.hf.BitsAndBytesConfig.reset_mock()
            self.hf.AutoConfig.from_pretrained.return_value.quantization_config = {"load_in_4bit": True}
            backend.load_model({**self.config, "source": "local", "model": str(output)})
            self.hf.BitsAndBytesConfig.assert_not_called()
            for loader in (self.hf.AutoConfig, self.hf.AutoTokenizer, self.hf.AutoModelForCausalLM):
                self.assertTrue(loader.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertNotIn("quantization_config", self.hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs)

    def test_failed_save_does_not_leave_partial_model(self):
        self.model.save_pretrained.side_effect = RuntimeError("disk full")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "saved"
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                backend.save_quantized_model(self.config, str(output))
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_local_path_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                backend.load_model({**self.config, "source": "local", "model": directory})
        self.hf.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_prequantized_hub_model_is_not_requantized(self):
        self.hf.AutoConfig.from_pretrained.return_value.quantization_config = {"quant_method": "gptq"}
        with self.assertRaisesRegex(ValueError, "이미 양자화"):
            backend.load_model(self.config)
        self.hf.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_cuda_and_dtype_validation(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            backend.load_model(self.config)
        self.torch.cuda.is_available.return_value = True
        with self.assertRaisesRegex(ValueError, "장치"):
            backend.load_model({**self.config, "device": "cuda:2"})
        self.torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            backend.load_model({**self.config, "compute_dtype": "bfloat16"})
        backend.load_model(self.config)
        self.assertEqual(self.hf.AutoModelForCausalLM.from_pretrained.call_args.kwargs["dtype"], self.torch.float16)

    def test_invalid_settings_fail_before_loading(self):
        invalid = {"source": "remote", "quantization": "Q4_K_M", "device": "cpu",
                   "compute_dtype": "int4", "temperature": -1, "max_new_tokens": 0,
                   "max_input_tokens": True, "enable_thinking": "false"}
        for key, value in invalid.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                backend.load_model({**self.config, key: value})
        self.hf.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_generation_decodes_only_new_tokens_and_preserves_messages(self):
        class Inputs(dict):
            def to(self, device):
                self.device = device
                return self

        inputs = Inputs(input_ids=SimpleNamespace(shape=(1, 12)), attention_mask="mask")
        self.tokenizer.apply_chat_template.return_value = inputs
        self.tokenizer.decode.return_value = " 답변 "
        messages = [{"role": "system", "content": "자료만 사용"}, {"role": "user", "content": "질문"}]
        answer = backend.generate_answer(messages, self.config)
        self.assertEqual(answer, "답변")
        self.assertEqual(inputs.device, "cuda:0")
        self.assertEqual(self.tokenizer.apply_chat_template.call_args.args[0], messages)
        self.assertFalse(self.model.generate.call_args.kwargs["do_sample"])
        self.assertNotIn("temperature", self.model.generate.call_args.kwargs)
        self.model.generate.return_value.__getitem__.assert_called_with((0, slice(12, None)))
        backend.generate_answer(messages, {**self.config, "temperature": 0.7})
        self.assertTrue(self.model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(self.model.generate.call_args.kwargs["temperature"], 0.7)
        self.model.generate.reset_mock()
        with self.assertRaisesRegex(ValueError, "max_input_tokens"):
            backend.generate_answer(messages, {**self.config, "max_input_tokens": 5})
        self.model.generate.assert_not_called()
        with self.assertRaisesRegex(ValueError, "컨텍스트"):
            backend.generate_answer(messages, {**self.config, "max_new_tokens": 8192})


if __name__ == "__main__":
    unittest.main()
