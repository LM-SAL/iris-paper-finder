"""
Used in the web app.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import spacy
from langchain_core.documents import Document

from paper_data_linking.process.embedders import ONNXEmbedder, Embedder
from paper_data_linking.process.plugins import Plugin, ZeroShotClassifier, is_soho_related
from paper_data_linking.process.splitters import PyMuPDFTokenSplitter, Splitter
from paper_data_linking.utils import ContentAnalyzerRecord, PluginRecord, get_stage_message

LOG = logging.getLogger(__name__)

NLP = spacy.blank("en")
NLP.add_pipe("sentencizer")


class ContentAnalyzer(ABC):
    def __init__(
        self,
        splitter: Splitter,
        embedder: Embedder,
        plugins: list[Plugin],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def process(self, content: Path | str):
        pass


class MyContentAnalyzer(ContentAnalyzer):
    def __init__(
        self,
        splitter: Splitter,
        embedder: Embedder,
        plugins: list[Plugin],
    ) -> None:
        self.splitter = splitter
        self.embedder = embedder
        self.validate_plugins(plugins)
        self.plugins = plugins

    @staticmethod
    def validate_plugins(plugins: list[Plugin]) -> None:
        """
        Validate the list of plugins.

        Specifically, make sure that 'embedder_kwargs["where"]' is a
        dictionary and that its keys are unique across plugins.
        """
        embedder_keys = set()

        for p in plugins:
            if "where" not in p.embedder_kwargs:
                msg = f"The plugin {p.name} does not have 'where' in embedder_kwargs."
                raise ValueError(msg)

            where_dict = p.embedder_kwargs["where"]

            if not isinstance(where_dict, dict):
                msg = f"'where' in plugin {p.name}'s embedder_kwargs should be a dictionary."
                raise TypeError(msg)

            for key in where_dict:
                if key in embedder_keys:
                    msg = f"Duplicate key '{key}' found in plugin {p.name}'s 'where' dictionary."
                    raise ValueError(msg)
                embedder_keys.add(key)

    @staticmethod
    def _get_full_text(docs, relevant_docs=None):
        if relevant_docs is None:
            relevant_docs = []
        relevant_positions = [d.metadata["position"] for d, _ in relevant_docs]
        full_text = ""
        for _i, d in enumerate(docs):
            pos = d.metadata["position"]
            content = d.page_content
            if pos in relevant_positions:
                content = f'<span style="background-color: #FFFF00">{content}</span>'
            txt = f"<h2>Excerpt {pos + 1}</h2>\n\n<blockquote><p>{content}</p></blockquote>"
            full_text += txt
        return full_text

    def enrich_metadata(self, docs, plugin):
        new_docs = []
        soho_related: list[bool] = []
        for d in docs:
            soho = is_soho_related(NLP, d.page_content, plugin.filter_terms, threshold=plugin.filter_threshold)
            key = next(iter(plugin.embedder_kwargs["where"].keys()))
            # assumes that first key is for heuristic filter
            # and that dict has ordered keys (true for python 3.7+)
            new_doc = Document(
                page_content=d.page_content,
                metadata=d.metadata | {key: int(soho)},
            )
            new_docs.append(new_doc)
            soho_related.append(soho)
        return new_docs, soho_related

    @staticmethod
    def _update_progress(stage: int, default_message: str, update_progress_func=None):
        """
        Internal method to streamline progress updates.
        """
        if update_progress_func:
            message = get_stage_message(stage)
            update_progress_func(message, default_message)

    def _apply_heuristic_filters(self, docs: list[Document]) -> tuple[list[Document], list[Plugin], list[PluginRecord]]:
        early_exit_records = []
        skip_plugins = []

        for p in self.plugins:
            docs, plugin_related = self.enrich_metadata(docs, p)
            if sum(plugin_related) == 0:
                skip_plugins.append(p)
                early_exit_record = p.create_early_exit_record()
                early_exit_records.append(early_exit_record)

        return docs, skip_plugins, early_exit_records

    def _process_plugins(
        self, docs: list[Document], skip_plugins: list[Plugin], update_progress=None
    ) -> list[PluginRecord]:
        all_records = []
        for i, p in enumerate(self.plugins):
            self._update_progress(1, f"Analyzing content (Plugin {i + 1}/{len(self.plugins)})", update_progress)
            if p in skip_plugins:
                continue  # skip plugins that did not pass heuristic filter

            analysis_record = p.process(docs, self.embedder)
            all_records.append(analysis_record)
        return all_records

    def process(self, content: Path | str, update_progress=None) -> ContentAnalyzerRecord:
        self._update_progress(-1, "Parsing and Splitting PDF content...", update_progress)
        LOG.info("Splitting")
        docs, ocr = self.splitter.split(content)
        LOG.info("Applying heuristic filters")
        docs, skip_plugins, early_exit_records = self._apply_heuristic_filters(docs)

        # Check if all plugins have exited early
        if len(skip_plugins) == len(self.plugins):
            all_records = early_exit_records
        else:
            self._update_progress(0, "Creating embeddings...", update_progress)
            LOG.info(f"Creating embeddings for {len(docs)} documents.")
            self.embedder.create_embeddings(docs)

            all_records = self._process_plugins(docs, skip_plugins, update_progress)
            all_records.extend(early_exit_records)  # Combine plugin records with early exit records
            # Not in original order,  but the records do have names.

        self._update_progress(2, "Finishing up...", update_progress)
        return ContentAnalyzerRecord(docs=docs, ocr_status=ocr, records=all_records)


def get_splitter(name):
    splitters = {
        "pymupdf_token": PyMuPDFTokenSplitter,
    }
    return splitters[name]


def get_embedder(name):
    embedders = {
        "onnx": ONNXEmbedder,
    }
    return embedders[name]


def get_plugin(name):
    plugins = {
        "SOHO_binary": ZeroShotClassifier,
    }
    return plugins[name]


def get_analyzer(config_paths, update_progress=None):
    splitter = PyMuPDFTokenSplitter(update_progress=update_progress)
    embedder = ONNXEmbedder()
    plugins = [ZeroShotClassifier.from_yaml(config_path) for config_path in config_paths]

    return MyContentAnalyzer(
        splitter=splitter,
        embedder=embedder,
        plugins=plugins,
    )
