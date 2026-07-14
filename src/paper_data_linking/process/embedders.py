"""Direct local ONNX embeddings and in-memory cosine ranking."""

# ruff: noqa: S101

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from functools import cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models/onnx"


class MiniLMModel:
    """The same all-MiniLM-L6-v2 inference used by Chroma's default embedder."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        missing = [str(path) for path in (model_path, tokenizer_path) if not path.is_file()]
        if missing:
            msg = f"Missing ONNX model files: {', '.join(missing)}. Run `make onnx`."
            raise FileNotFoundError(msg)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)  # noqa: S106
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch(texts[start : start + batch_size])
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            output = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": np.zeros_like(input_ids),
                },
            )[0]
            mask = attention_mask[..., None]
            embeddings = (output * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            batches.append((embeddings / np.clip(norms, 1e-12, None)).astype(np.float32))
        return np.concatenate(batches)


@cache
def _load_model(model_dir: str) -> MiniLMModel:
    return MiniLMModel(model_dir)


class Embedder(ABC):
    @abstractmethod
    def create_embeddings(self, docs: list) -> None:
        pass

    @abstractmethod
    def get_relevant_docs(self, query: str, kwargs: dict | None = None, n_results: int = 10):
        pass


class ONNXEmbedder(Embedder):
    """Keep documents and normalized embeddings in memory for one paper."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model = _load_model(str(Path(model_dir).resolve()))
        self.docs = []
        self.embeddings = np.empty((0, 384), dtype=np.float32)

    def create_embeddings(self, docs: list) -> None:
        self.docs = list(docs)
        self.embeddings = self.model.encode([doc.page_content for doc in self.docs])

    def get_relevant_docs(self, query: str, kwargs: dict | None = None, n_results: int = 10):
        if n_results <= 0:
            return [], []
        kwargs = kwargs or {}
        unsupported = set(kwargs) - {"where"}
        if unsupported:
            msg = f"Unsupported ranking options: {sorted(unsupported)}"
            raise ValueError(msg)
        where = kwargs.get("where", {})
        indices = [
            index
            for index, doc in enumerate(self.docs)
            if all(doc.metadata.get(key) == value for key, value in where.items())
        ]
        if not indices:
            return [], []
        query_embedding = self.model.encode([query])[0]
        distances = 1.0 - self.embeddings[indices] @ query_embedding
        order = np.argsort(distances, kind="stable")[:n_results]
        return (
            [self.docs[indices[index]] for index in order],
            [float(distances[index]) for index in order],
        )


def _self_check() -> None:
    class Doc:
        def __init__(self, name: str, group: int) -> None:
            self.page_content = name
            self.metadata = {"name": name, "group": group}

    embedder = ONNXEmbedder.__new__(ONNXEmbedder)
    embedder.docs = [Doc("far", 1), Doc("best", 1), Doc("filtered", 0)]
    embedder.embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    class FakeModel:
        @staticmethod
        def encode(_texts: list[str]) -> np.ndarray:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    embedder.model = FakeModel()
    docs, distances = embedder.get_relevant_docs("query", {"where": {"group": 1}})
    assert [doc.metadata["name"] for doc in docs] == ["best", "far"]
    assert distances == [0.0, 1.0]


if __name__ == "__main__":
    _self_check()
