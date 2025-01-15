import pytest
from unittest.mock import Mock, patch
from paper_data_linking.process.analyzer import MyContentAnalyzer, PyMuPDFTokenSplitter, ChromaEmbedder
from langchain.schema import Document


@pytest.fixture
def mock_plugin():
    plugin = Mock()
    plugin.name = "mock_plugin"
    plugin._answer_key = "mock_answer"
    plugin.process.return_value = ({"key": "value"}, "mock_analysis", [0])
    plugin.embedder_kwargs = {"where": {"passed_heuristic": 1}}
    plugin.filter_terms = ["sun", "golden"]
    plugin.filter_threshold = 0.8
    return plugin


@pytest.fixture
def mock_embedder():
    embedder = Mock(spec=ChromaEmbedder)
    return embedder


@pytest.fixture
def mock_splitter():
    splitter = Mock(spec=PyMuPDFTokenSplitter)

    # Mock the split method of splitter
    mock_docs = [
        Document(
            page_content="Golden morning light,",
            metadata={"position": 1}
        ),
        Document(
            page_content="Shadows fade from sight,",
            metadata={"position": 2}
        ),
        Document(
            page_content="Sunrise warms the sky.",
            metadata={"position": 3}
        ),
    ]
    mock_ocr_status = False  # or True, based on your test scenario

    splitter.split.return_value = (mock_docs, mock_ocr_status)
    return splitter


@pytest.fixture
def content_analyzer(mock_splitter, mock_plugin, mock_embedder):
    return MyContentAnalyzer(
        splitter=mock_splitter,
        embedder=mock_embedder,
        plugins=[mock_plugin]
    )


def test_process_all_plugins_pass(content_analyzer):
    result = content_analyzer.process(b"mock_content")
    assert result.ocr_status is not None
    assert "mock_plugin" in result.records
    assert result.records["mock_plugin"].data == {"key": "value"}


def test_process_some_plugins_skipped(content_analyzer, mock_plugin):
    # Mock the enrich_metadata method to simulate skipping the plugin
    with patch.object(content_analyzer, "enrich_metadata", return_value=([], [0])):
        result = content_analyzer.process("mock_content")
    assert result.ocr_status is not None
    assert "mock_plugin" in result.records


def test_process_all_plugins_skipped(content_analyzer, mock_plugin):
    # Mock the enrich_metadata method to simulate skipping all plugins
    with patch.object(content_analyzer, "enrich_metadata", return_value=([], [0])):
        result = content_analyzer.process("mock_content")
    assert result.ocr_status is not None

