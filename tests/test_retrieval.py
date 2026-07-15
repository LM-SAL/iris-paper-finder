from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from iris_paper_llm.models import PaperChunk, RetrievalMode, SelectionReason
from iris_paper_llm.retrieval import (
    ONNXEmbedder,
    is_exact_iris_match,
    remove_reference_section,
    retrieve_chunks,
)


class FakeEmbedder:
    def create_embeddings(self, docs: list) -> None:
        self.docs = docs

    def get_relevant_docs(self, _query: str, *, top_k: int = 10, where: dict | None = None):
        docs = self.docs
        if where:
            docs = [doc for doc in docs if doc.metadata["is_candidate"] == 1]
        docs = sorted(docs, key=lambda doc: doc.metadata["position"], reverse=True)[:top_k]
        return docs, [float(index) / 10 for index in range(len(docs))]


def test_retrieval_modes_and_matching() -> None:
    chunks = [
        PaperChunk(page=1, chunk_id="p1-c0", text="Introduction without a mission name."),
        PaperChunk(page=1, chunk_id="p1-c1", text="We analyze IRIS observations."),
        PaperChunk(page=2, chunk_id="p2-c0", text="The result is shown here."),
        PaperChunk(page=2, chunk_id="p2-c1", text="Unrelated appendix."),
    ]
    result = retrieve_chunks(chunks, top_k=4, embedder=FakeEmbedder())
    assert [chunk.chunk_id for chunk in result.selected] == ["p2-c0", "p1-c1", "p1-c0", "p2-c1"]
    assert result.selected[-1].reason == SelectionReason.GLOBAL_FALLBACK
    assert is_exact_iris_match("Synthetic Si IV at 1402.8 Angstrom")
    assert not is_exact_iris_match("Synthetic Fe XII at 195 Angstrom")
    assert remove_reference_section(["Methods\nsources of data", "Results", "References\nCitation"])[-1] == "Results"
    assert remove_reference_section(["Introduction\nreferences therein", "Results"])[-1] == "Results"

    no_match = retrieve_chunks(
        [PaperChunk(page=1, chunk_id="only", text="No relevant term here.")],
        embedder=FakeEmbedder(),
    )
    assert [chunk.chunk_id for chunk in no_match.selected] == ["only"]

    heuristic = retrieve_chunks(chunks, mode=RetrievalMode.HEURISTIC, top_k=4, embedder=FakeEmbedder())
    assert len(heuristic.selected) == 3
    assert SelectionReason.GLOBAL_FALLBACK not in {chunk.reason for chunk in heuristic.selected}

    all_chunks = retrieve_chunks(chunks, mode=RetrievalMode.ALL, top_k=2, embedder=FakeEmbedder())
    assert all(chunk.reason == SelectionReason.GLOBAL_FALLBACK for chunk in all_chunks.selected)


def test_numpy_ranking() -> None:
    embedder = ONNXEmbedder.__new__(ONNXEmbedder)
    embedder.docs = [
        SimpleNamespace(page_content="far", metadata={"name": "far", "group": 1}),
        SimpleNamespace(page_content="best", metadata={"name": "best", "group": 1}),
        SimpleNamespace(page_content="filtered", metadata={"name": "filtered", "group": 0}),
    ]
    embedder.embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    embedder.model = SimpleNamespace(
        encode=lambda _texts: np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    docs, distances = embedder.get_relevant_docs("query", where={"group": 1})
    assert [doc.metadata["name"] for doc in docs] == ["best", "far"]
    assert distances == [0.0, 1.0]


def test_model_directory_environment_override() -> None:
    with (
        patch.dict("os.environ", {"IRIS_PAPER_LLM_MODEL_DIR": "/tmp/custom-model"}),
        patch("iris_paper_llm.retrieval._load_model") as load_model,
    ):
        ONNXEmbedder()
    load_model.assert_called_once_with("/tmp/custom-model")


if __name__ == "__main__":
    test_retrieval_modes_and_matching()
    test_numpy_ranking()
    test_model_directory_environment_override()
