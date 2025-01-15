from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class PhenomenaExtractor:

    def __init__(self, model="gpt-3.5-turbo", temperature=0):
        self.model = ChatOpenAI(model=model, temperature=temperature)

    def _get_relevant_chunks(self, db, k=6):
        query = "What phenomena are discussed?"
        data_docs = db.similarity_search_with_score(query, k=k)
        return data_docs

    def _create_context(self, data_docs):
        """
        Concatenates all document contents into a single text.

        Args:
            data_docs (list): List of tuples. Each tuple contains a document
                              and a score.

        Returns:
            full_text (str): Concatenated document contents.
        """

        full_text = ""
        for i, (dd, score) in enumerate(data_docs):
            text = dd.page_content
            full_text += f"\n\n#Excerpt {i + 1}##\n\n{text}"
        return full_text

    def _create_messages(self, full_text):
        """
        Creates the list of messages for the language model.

        Args:
            full_text (str): Concatenated document contents.

        Returns:
            messages (list): List of messages for the language model.
        """

        messages = [
            SystemMessage(
                content="""You are a helpful assistant who helps analyze scientific documents and you produce Markdown outputs."""
            ),
            HumanMessage(
                content=f""""```{full_text}```\n\n Based on the excerpts provided, please provide a list in JSON of all phenomena detailed in this paper (ex. ["phenomena 1", "phenomena 2"]):"""
            ),
        ]
        return messages

    def run_llm_chain(self, db):
        """
        Analyzes the given documents and determines whether they are
        SOHO mission papers.

        Args:
            data_docs (list): List of tuples. Each tuple contains a langchain document
                              and a score.

        Returns:
            result: Result of the analysis by the language model.
        """

        data_docs = self._get_relevant_chunks(db)
        full_text = self._create_context(data_docs)
        messages = self._create_messages(full_text)
        result = self.model(messages)
        return result, data_docs


class InstrumentExtractor:

    def __init__(self, model="gpt-3.5-turbo", temperature=0):
        self.model = ChatOpenAI(model=model, temperature=temperature)

    def _get_relevant_chunks(self, db, k=6):
        query = "Does this paper use data from the SOHO spacecraft or its instruments (CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO)?"
        data_docs = db.similarity_search_with_score(query, k=k)
        return data_docs

    def _create_context(self, data_docs):
        """
        Concatenates all document contents into a single text.

        Args:
            data_docs (list): List of tuples. Each tuple contains a document
                              and a score.

        Returns:
            full_text (str): Concatenated document contents.
        """

        full_text = ""
        for i, (dd, score) in enumerate(data_docs):
            text = dd.page_content
            full_text += f"\n\n#Excerpt {i + 1}##\n\n{text}"
        return full_text

    def _create_messages(self, full_text):
        """
        Creates the list of messages for the language model.

        Args:
            full_text (str): Concatenated document contents.

        Returns:
            messages (list): List of messages for the language model.
        """

        messages = [
            SystemMessage(
                content="""You are a helpful assistant who helps analyze scientific documents and you produce Markdown outputs."""
            ),
            HumanMessage(
                content=f""""```{full_text}```\n\n Based on the excerpts provided, please provide a list in JSON of all SOHO instruments used in this paper (ex. ["instrument 1", "instrument 2"]):

    For your reference, here is a list of SOHO instruments: CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO."""
            ),
        ]
        return messages

    def run_llm_chain(self, db):
        """
        Analyzes the given documents and determines whether they are
        SOHO mission papers.

        Args:
            data_docs (list): List of tuples. Each tuple contains a langchain document
                              and a score.

        Returns:
            result: Result of the analysis by the language model.
        """

        data_docs = self._get_relevant_chunks(db)
        full_text = self._create_context(data_docs)
        messages = self._create_messages(full_text)
        result = self.model(messages)
        return result, data_docs


class SOHOIdentifier:

    def __init__(self, model="gpt-3.5-turbo", temperature=0):
        self.model = ChatOpenAI(model=model, temperature=temperature)

    def _get_relevant_chunks(self, db, k=6):
        query = "Does this paper use data from the SOHO spacecraft or its instruments (CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO)?"
        data_docs = db.similarity_search_with_score(query, k=k)
        return data_docs

    def _create_context(self, data_docs):
        """
        Concatenates all document contents into a single text.

        Args:
            data_docs (list): List of tuples. Each tuple contains a document
                              and a score.

        Returns:
            full_text (str): Concatenated document contents.
        """

        full_text = ""
        for i, (dd, score) in enumerate(data_docs):
            text = dd.page_content
            full_text += f"\n\n#Excerpt {i + 1}##\n\n{text}"
        return full_text

    def _create_messages(self, full_text):
        """
        Creates the list of messages for the language model.

        Args:
            full_text (str): Concatenated document contents.

        Returns:
            messages (list): List of messages for the language model.
        """

        messages = [
            SystemMessage(
                content="""You are a helpful assistant who helps analyze scientific documents and you produce Markdown outputs."""
            ),
            HumanMessage(
                content=f""""```{full_text}```\n\n Based on the excerpts provided, please answer the following questions:

A. Does this paper directly use data from the SOHO mission and its primary catalogues?
B. Does this paper make quantitative predictions of results from the SOHO mission?
C. Does this paper describe the SOHO mission, its instruments, operations, software, or calibrations?

For your reference, here is a list of SOHO instruments: CDS, CELIAS, COSTEP, EIT, ERNE, GOLF, LASCO, MDI, SUMER, SWAN, UVCS, VIRGO.

If the answer to ANY of these questions is yes (even a little bit), the paper is considered a SOHO paper. Please go through each excerpt and provide reasoning for what each excerpt tells us. Finally, conclude whether the overall paper can be classified as a SOHO paper."""
            ),
        ]
        return messages

    def run_llm_chain(self, db):
        """
        Analyzes the given documents and determines whether they are
        SOHO mission papers.

        Args:
            data_docs (list): List of tuples. Each tuple contains a langchain document
                              and a score.

        Returns:
            result: Result of the analysis by the language model.
        """

        data_docs = self._get_relevant_chunks(db)
        full_text = self._create_context(data_docs)
        messages = self._create_messages(full_text)
        result = self.model(messages)
        return result, data_docs
