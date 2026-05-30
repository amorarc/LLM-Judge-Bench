from __future__ import annotations

import logging
from typing import List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

_threads_configured = False

# Registry of named model classes (besides the default AutoModelForCausalLM).
_MODEL_CLASS_REGISTRY = {
    "mistral3": ("transformers", "Mistral3ForConditionalGeneration"),
}


class HuggingFaceModel(BaseModel):
    """
    Causal-LM wrapper using HuggingFace Transformers.

    S[i, j] = (1/T_j) * Σ_t log p(x_{j,t} | x_{j,<t})

    Model is loaded lazily on first call.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "float32",
        batch_size: int = 8,
        max_length: int = 512,
        num_threads: int = 4,
        quantization: Optional[str] = None,
        model_class: str = "auto",
    ):
        super().__init__(model_id)
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._dtype = dtype
        self._num_threads = num_threads
        self._quantization = quantization  # "int8" | "int4" | "fp8_fg" | None
        self._model_class = model_class    # "auto" | "mistral3"
        self._loaded = False

    def _build_quant_config(self):
        if self._quantization == "int8":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(load_in_8bit=True)
        if self._quantization == "int4":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._requested_dtype,
            )
        if self._quantization == "fp8_fg":
            from transformers import FineGrainedFP8Config
            return FineGrainedFP8Config(dequantize=True)
        return None

    def _resolve_model_class(self):
        if self._model_class == "auto":
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM
        if self._model_class not in _MODEL_CLASS_REGISTRY:
            raise ValueError(
                f"Unknown model_class '{self._model_class}'. "
                f"Valid options: auto, {', '.join(_MODEL_CLASS_REGISTRY)}"
            )
        module_name, cls_name = _MODEL_CLASS_REGISTRY[self._model_class]
        import importlib
        return getattr(importlib.import_module(module_name), cls_name)

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer

        logger.info(f"Loading {self.model_id} …")
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            # fp8 — PyTorch >= 2.1, H100/Ada GPU required
            "float8":        getattr(torch, "float8_e4m3fn", None),
            "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
            "float8_e5m2":   getattr(torch, "float8_e5m2",   None),
        }
        requested = dtype_map.get(self._dtype)
        if requested is None:
            if self._dtype.startswith("float8"):
                raise RuntimeError(
                    f"dtype '{self._dtype}' requires PyTorch >= 2.1 and an H100/Ada GPU "
                    f"(current torch: {torch.__version__})"
                )
            requested = torch.float32
        self._requested_dtype = requested

        self._torch = torch
        global _threads_configured
        if not _threads_configured:
            torch.set_num_threads(self._num_threads)
            torch.set_num_interop_threads(max(1, self._num_threads // 2))
            _threads_configured = True
        device_map = "cuda" if torch.cuda.is_available() else self._device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = self._build_quant_config()
        load_kwargs: dict = {"device_map": device_map}
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            # int8 MatMul8bitLt only supports float16; bfloat16 causes CUBLAS errors
            load_kwargs["dtype"] = torch.float16 if self._quantization == "int8" else requested
            logger.info(f"  quantization: {self._quantization}")
        else:
            load_kwargs["dtype"] = requested
            logger.info(f"  dtype: {self._dtype}")

        model_cls = self._resolve_model_class()
        logger.info(f"  model class: {model_cls.__name__}")
        self.model = model_cls.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self._loaded = True
        logger.info(f"Loaded {self.model_id}")

    def unload(self) -> None:
        if not self._loaded:
            return
        del self.model
        del self.tokenizer
        self._loaded = False
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info(f"Unloaded {self.model_id}")

    def logprobs(self, text: str) -> float:
        return self.batch_logprobs([text])[0]

    def batch_logprobs(self, texts: List[str]) -> List[float]:
        from tqdm import tqdm
        self._load()
        results: list[float] = []
        chunks = range(0, len(texts), self.batch_size)
        bar = tqdm(chunks, desc=self.model_id.split("/")[-1], unit="batch",
                   total=len(chunks), leave=False)
        for i in bar:
            results.extend(self._score_chunk(texts[i : i + self.batch_size]))
        return results

    def _score_chunk(self, texts: list[str]) -> list[float]:
        torch = self._torch
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Shift: predict token t from context t-1
        token_lp = log_probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = attention_mask[:, 1:].float()
        mean_lp = (token_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return mean_lp.cpu().tolist()
