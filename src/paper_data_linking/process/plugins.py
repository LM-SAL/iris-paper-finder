import json
import logging
from abc import ABC, abstractmethod
from json.decoder import JSONDecodeError

import openai
import spacy
import yaml
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from thefuzz import fuzz

from paper_data_linking.process.embedders import Embedder
from paper_data_linking.settings import OPENAI_API_KEY
from paper_data_linking.utils import PluginRecord, get_correct_instruments

LOG = logging.getLogger(__name__)
openai.api_key = OPENAI_API_KEY


class Plugin(ABC):
    def __init__(self) -> None:
        self._query = None

    @abstractmethod
    def process(
        self, docs: list[Document], embedder: Embedder, embedder_kwargs: dict | None = None
    ) -> tuple[dict, str, list[Document]]:
        pass


class SOHOStepwiseBinaryClassifier(Plugin):
    def __init__(self, kwargs) -> None:
        super().__init__()
        self._query = "Does this paper use data from the SOHO spacecraft or its instruments (CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO)?"
        self.system_message = SystemMessage(
            content="""You are a helpful assistant who analyzes scientific documents and you produce Markdown outputs."""
        )
        self.model = ChatOpenAI(model=kwargs["model"], temperature=kwargs["temperature"])
        # ^ should probably just unpack these with **kwargs
        self.questions = [
            "Do these excerpts directly use data from the SOHO mission and its primary catalogues?",
            "Do these excerpts make quantitative predictions of results from the SOHO mission?",
            "Do these excerpts describe the SOHO mission, its instruments, operations, software, or calibrations?",
        ]

    @staticmethod
    def _construct_question(context, question):
        return f"""For your reference, the Solar and Heliospheric Observatory (SOHO) has these instruments: CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO.

        Now, use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

    def _get_questions(self, context):
        questions = []
        for q in self.questions:
            message = HumanMessage(content=self._construct_question(context, q))
            questions.append(message)
        return questions

    def _ask_questions(self, docs):
        context = self._create_context(docs)
        questions = self._get_questions(context)
        results = []
        for q in questions:
            messages = [self.system_message, q]
            result = self.model.invoke(messages)
            results.append(result)
        return results

    @staticmethod
    def _create_context(docs: list[Document]) -> str:
        full_text = ""
        for i, (dd, _score) in enumerate(docs):
            text = dd.page_content
            full_text += f"{i + 1}. {text}\n\n"
        return full_text

    def _get_second_human_message(self, analysis):
        return HumanMessage(
            content=f"""```{analysis}```
Here are the answers to three questions. Is the answer to ANY of these questions yes? Answer with YES, NO, or UNCERTAIN. Give no explanation. Just one word."""
        )

    def _get_second_set_of_messages(self, analysis):
        return [self.system_message, self._get_second_human_message(analysis)]

    def process(
        self,
        docs: list[Document],  # NOQA: ARG002
        embedder: Embedder,
        embedder_kwargs: dict | None = None,
    ) -> tuple[dict, str, list[Document]]:
        relevant_docs = embedder.get_relevant_docs(self._query, embedder_kwargs)
        results = self._ask_questions(relevant_docs)
        analyses = []
        for _, (q, result) in enumerate(zip(self.questions, results, strict=False)):
            analysis = f"#### {q}\n" + result.content
            analyses.append(analysis)
        total_analysis = "\n\n".join(analyses)

        second_messages = self._get_second_set_of_messages(total_analysis)
        answer_result = self.model(second_messages)

        data = {"SOHO": answer_result.content}
        return data, total_analysis, relevant_docs


def get_remaining_contents(text, phrase):
    try:
        return text.split(phrase)[1].split("```")[0]
    except Exception:
        LOG.exception("Error in get_remaining_contents")
        return None


def get_contents_after_phrase_before_newline(text, phrase):
    # Look for the string 'Classification:' in the text
    start = text.rfind(phrase)
    if start != -1:
        # If found, extract the remainder of the line
        end = text.find("\n", start)
        if end == -1:
            end = len(text)
        classification_line = text[start:end].strip()
        # Split the line on ':', and return the second part (the classification value)
        _, value = classification_line.split(":", 1)
        return value.strip()
    return None


def get_dois(instruments: list, metadata: dict):
    if metadata is None:
        return [{"label": inst, "link": "not found"} for inst in instruments]
    instruments_data = []
    for inst in instruments:
        doi = metadata[inst]["link"] if inst in metadata else "not found"
        inst_data = {"label": inst, "link": doi}
        instruments_data.append(inst_data)
    return instruments_data


def is_soho_related(nlp, txt, indicator_strings, threshold=80):
    doc = nlp(txt)
    for sentence in doc.sents:
        for si in indicator_strings:
            if si in sentence.text:
                return True
            if fuzz.partial_ratio(si, sentence.text) > threshold:
                return True
    return False


class ZeroShotClassifier(Plugin):
    def __init__(self, nl_config, model_kwargs, embedder_kwargs=None) -> None:
        super().__init__()
        self._load_config_from_dict(nl_config)
        self.validate_model_kwargs(model_kwargs)
        self.model = ChatOpenAI(**model_kwargs)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.embedder_kwargs = embedder_kwargs

    # method which validates that model_kwargs has 'model_name' and 'temperature' keys
    def validate_model_kwargs(self, model_kwargs):
        if "model_name" not in model_kwargs or "temperature" not in model_kwargs:
            msg = "Invalid model_kwargs configuration: Missing required fields."
            raise ValueError(msg)

    def _load_config_from_dict(self, config):
        self.name = config.get("name")
        self._query = config.get("query")
        self.system_message = SystemMessage(content=config.get("system_message"))
        self._human_message = config.get("human_message")
        self._answer_divider = config.get("answer_divider")
        self._json_divider = config.get("json_divider")
        self._answer_key = config.get("answer_key")
        self._json_key = config.get("json_key")
        self.label_metadata_map = config.get("label_metadata_map")
        self.filter_terms = config.get("filter_terms")
        self.filter_threshold = config.get("filter_threshold")

    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract and validate classifier config
        classifier_config = config.get("classifier", {})
        if (
            "name" not in classifier_config
            or "query" not in classifier_config
            or "system_message" not in classifier_config
            or "human_message" not in classifier_config
            or "answer_divider" not in classifier_config
            or "json_divider" not in classifier_config
            or "answer_key" not in classifier_config
            or "json_key" not in classifier_config
            or "label_metadata_map" not in classifier_config
            or "filter_terms" not in classifier_config
            or "filter_threshold" not in classifier_config
        ):
            msg = "Invalid classifier configuration in YAML: Missing required fields."
            raise ValueError(msg)

        # Extract model_kwargs and embedder_kwargs
        model_kwargs = config.get("model_kwargs", {})
        embedder_kwargs = config.get("embedder_kwargs", {})

        return cls(classifier_config, model_kwargs, embedder_kwargs)

    def _get_human_message(self, context):
        return HumanMessage(content=f"```{context}```\n\n {self._human_message}")

    @staticmethod
    def _create_context(docs: list[Document]) -> str:
        full_text = ""
        for i, dd in enumerate(docs):
            text = dd.page_content
            p = dd.metadata.get("position", i)
            full_text += f"\n\n#Excerpt {p + 1}##\n\n{text}"
        return full_text

    def _get_messages(self, docs):
        context = self._create_context(docs)
        return [self.system_message, self._get_human_message(context)]

    def get_answer_from_analysis(self, text):
        return get_contents_after_phrase_before_newline(text, phrase=self._answer_divider)

    def get_instruments_from_analysis(self, text):
        inst_str = get_remaining_contents(text, phrase=self._json_divider)
        if inst_str is not None:
            instruments = json.loads(inst_str)
            instruments = get_correct_instruments(instruments, metadata=self.label_metadata_map)
            instruments_data = get_dois(instruments, metadata=self.label_metadata_map)
        else:
            instruments_data = []
        return instruments_data

    def create_early_exit_record(self) -> PluginRecord:
        return PluginRecord(
            analyzer=self.name,
            passed_heuristic_filter=False,
            data=None,
            analysis=None,
            relevant_indices=None,
        )

    def process(self, docs: list[Document], embedder: Embedder) -> PluginRecord:
        relevant_docs, _distances = embedder.get_relevant_docs(self._query, self.embedder_kwargs)
        LOG.info(f"Relevant docs: {len(relevant_docs)}")
        messages = self._get_messages(relevant_docs)
        LOG.info("Querying LLM")
        result = self.model.invoke(messages)
        analysis = result.content
        LOG.info("Got analysis from LLM")
        try:
            answer = self.get_answer_from_analysis(analysis)
        except JSONDecodeError:
            LOG.info("Could not decode JSON from analysis.")
            answer = ""
        try:
            instruments = self.get_instruments_from_analysis(analysis)
        except JSONDecodeError:
            LOG.info("Could not decode JSON from analysis.")
            instruments = []
        data = {
            self._answer_key: answer,
            self._json_key: instruments,
            "analyzer": self.name,
        }
        relevant_indices = []
        LOG.info("Getting relevant indices")
        for d in relevant_docs:
            try:
                i = docs.index(d)
                relevant_indices.append(i)
            except ValueError:
                LOG.info("Could not find relevant doc in original list.")
        return PluginRecord(
            analyzer=self.name,
            passed_heuristic_filter=True,
            data=data,
            analysis=analysis,
            relevant_indices=relevant_indices,
        )
