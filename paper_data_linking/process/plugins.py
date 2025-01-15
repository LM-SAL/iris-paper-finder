import json
import logging
from typing import List
import random
from json.decoder import JSONDecodeError
from abc import ABC, abstractmethod
import yaml
from typing import Tuple

import openai
import spacy
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document, SystemMessage, HumanMessage
from thefuzz import fuzz

from paper_data_linking.process.embedders import Embedder
from paper_data_linking.settings import OPENAI_API_KEY
from paper_data_linking.utils import get_correct_instruments, PluginRecord

LOG = logging.getLogger(__file__)
openai.api_key = OPENAI_API_KEY


class Plugin(ABC):
    def __init__(self):
        self._query = None

    @abstractmethod
    def process(
        self, docs: list[Document], embedder: Embedder, embedder_kwargs: dict = None
    ) -> Tuple[dict, str, list[Document]]:
        pass


class FakePlugin(Plugin):
    def __init__(self):
        self._query = "Does this paper use data from the SOHO spacecraft or its instruments (CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO)?"

    def process(
        self, docs: list[Document], embedder: Embedder, embedder_kwargs: dict = None
    ) -> Tuple[dict, str, list[Document]]:
        relevant_docs = embedder.get_relevant_docs(self._query, embedder_kwargs)
        analysis = "This paper looks like it is pretty cool."
        soho = random.choice(["YES", "NO", "UNCERTAIN"])
        data = {"status": "very cool", "space": True, "SOHO": soho}
        return data, analysis, relevant_docs


class SOHOStepwiseBinaryClassifier(Plugin):
    def __init__(self, kwargs):
        super().__init__()
        self._query = "Does this paper use data from the SOHO spacecraft or its instruments (CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO)?"
        self.system_message = SystemMessage(
            content="""You are a helpful assistant who analyzes scientific documents and you produce Markdown outputs."""
        )
        self.model = ChatOpenAI(
            model=kwargs["model"], temperature=kwargs["temperature"]
        )
        # ^ should probably just unpack these with **kwargs
        self.questions = [
            "Do these excerpts directly use data from the SOHO mission and its primary catalogues?",
            "Do these excerpts make quantitative predictions of results from the SOHO mission?",
            "Do these excerpts describe the SOHO mission, its instruments, operations, software, or calibrations?",
        ]

    @staticmethod
    def _construct_question(context, question):
        p = f"""For your reference, the Solar and Heliospheric Observatory (SOHO) has these instruments: CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO. 
        
        Now, use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
        return p

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
            result = self.model(messages)
            results.append(result)
        return results

    @staticmethod
    def _create_context(docs: list[Document]) -> str:
        full_text = ""
        for i, (dd, score) in enumerate(docs):
            text = dd.page_content
            full_text += f"{i + 1}. {text}\n\n"
        return full_text

    def _get_second_human_message(self, analysis):
        human_message = HumanMessage(
            content=f"""```{analysis}```
Here are the answers to three questions. Is the answer to ANY of these questions yes? Answer with YES, NO, or UNCERTAIN. Give no explanation. Just one word."""
        )
        return human_message

    def _get_second_set_of_messages(self, analysis):
        messages = [self.system_message, self._get_second_human_message(analysis)]
        return messages

    def process(
        self, docs: list[Document], embedder: Embedder, embedder_kwargs: dict = None
    ) -> Tuple[dict, str, list[Document]]:
        relevant_docs = embedder.get_relevant_docs(self._query, embedder_kwargs)
        # analyses = []
        # for i, d in enumerate(relevant_docs):

        results = self._ask_questions(relevant_docs)
        print(results)

        analyses = []
        for i, (q, result) in enumerate(zip(self.questions, results)):
            analysis = f"#### {q}\n" + result.content
            analyses.append(analysis)
        total_analysis = "\n\n".join(analyses)

        second_messages = self._get_second_set_of_messages(total_analysis)
        answer_result = self.model(second_messages)

        data = {"SOHO": answer_result.content}
        return data, total_analysis, relevant_docs


def get_remaining_contents(text, phrase):
    value = text.split(phrase)[1].split("```")[0]
    return value


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
    else:
        return None

def get_dois(instruments: list, metadata: dict):
    if metadata is None:
        return [{"label": inst, "link": "not found"} for inst in instruments]
    instruments_data = []
    for inst in instruments:
        if inst in metadata:
            doi = metadata[inst]['link']
        else:
            doi = "not found"
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
    def __init__(self, nl_config, model_kwargs, embedder_kwargs=None):
        super().__init__()
        self._load_config_from_dict(nl_config)
        self.validate_model_kwargs(model_kwargs)
        self.model = ChatOpenAI(**model_kwargs)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.embedder_kwargs = embedder_kwargs

    # method which validates that model_kwargs has 'model_name' and 'temperature' keys
    def validate_model_kwargs(self, model_kwargs):
        if not "model_name" in model_kwargs or not "temperature" in model_kwargs:
            raise ValueError(
                "Invalid model_kwargs configuration: Missing required fields."
            )

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
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract and validate classifier config
        classifier_config = config.get("classifier", {})
        if (
            not "name" in classifier_config
            or not "query" in classifier_config
            or not "system_message" in classifier_config
            or not "human_message" in classifier_config
            or not "answer_divider" in classifier_config
            or not "json_divider" in classifier_config
            or not "answer_key" in classifier_config
            or not "json_key" in classifier_config
            or not "label_metadata_map" in classifier_config
            or not "filter_terms" in classifier_config
            or not "filter_threshold" in classifier_config
        ):
            raise ValueError("Invalid classifier configuration in YAML: Missing required fields.")

        # Extract model_kwargs and embedder_kwargs
        model_kwargs = config.get("model_kwargs", {})
        embedder_kwargs = config.get("embedder_kwargs", {})

        return cls(classifier_config, model_kwargs, embedder_kwargs)

    def _get_human_message(self, context):
        human_message = HumanMessage(
            content=f"```{context}```\n\n {self._human_message}"
        )
        return human_message

    @staticmethod
    def _create_context(docs: list[Document]) -> str:
        full_text = ""
        for i, dd in enumerate(docs):
            text = dd.page_content
            if "position" in dd.metadata:
                p = dd.metadata["position"]
            else:
                p = i
            full_text += f"\n\n#Excerpt {p + 1}##\n\n{text}"
        return full_text

    def _get_messages(self, docs):
        context = self._create_context(docs)
        messages = [self.system_message, self._get_human_message(context)]
        return messages

    def get_answer_from_analysis(self, text):
        val = get_contents_after_phrase_before_newline(text, phrase=self._answer_divider)
        return val

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

    def process(self, docs: List[Document], embedder: Embedder) -> PluginRecord:
        relevant_docs, distances = embedder.get_relevant_docs(self._query, self.embedder_kwargs)
        LOG.warning(f"Relevant docs: {len(relevant_docs)}")
        messages = self._get_messages(relevant_docs)
        LOG.warning(f"Querying LLM")
        result = self.model(messages)
        analysis = result.content
        LOG.warning(f"Got analysis from LLM")
        try:
            answer = self.get_answer_from_analysis(analysis)
        except JSONDecodeError:
            LOG.warning("Could not decode JSON from analysis.")
            answer = ""
        try:
            instruments = self.get_instruments_from_analysis(analysis)
        except JSONDecodeError:
            LOG.warning("Could not decode JSON from analysis.")
            instruments = []
        data = {
            self._answer_key: answer,
            self._json_key: instruments,
            "analyzer": self.name,
        }
        relevant_indices = []
        LOG.warning(f"Getting relevant indices")
        for d in relevant_docs:
            try:
                i = docs.index(d)
                relevant_indices.append(i)
            except ValueError:
                LOG.warning("Could not find relevant doc in original list.")
        # relevant_indices = [docs.index(d) for d in relevant_docs]
        return PluginRecord(
            analyzer=self.name,
            passed_heuristic_filter=True,
            data=data,
            analysis=analysis,
            relevant_indices=relevant_indices,
        )