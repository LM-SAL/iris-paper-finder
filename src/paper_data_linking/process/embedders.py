import logging
import os
import random
import uuid
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

from paper_data_linking.utils import clean_collection_name_for_chroma

LOG = logging.getLogger(__file__)


class Embedder(ABC):
    @abstractmethod
    def create_embeddings(self, docs: List[Document]):
        pass

    @staticmethod
    def query(self, docs: List[Document]):
        pass

    @abstractmethod
    def load_embeddings(self, in_embeddings: os.PathLike):
        pass

    @abstractmethod
    def get_relevant_docs(
        self, query, kwargs: Union[None, dict] = None
    ) -> Tuple[List[Document], List[float]]:
        pass


class FakeEmbedder(Embedder):
    def __init__(self):
        self._docs = None

    def create_embeddings(self, docs: List[Document]):
        embeddings = [[random.random() for _ in range(2)] for _ in docs]
        self._docs = docs
        return embeddings

    def get_relevant_docs(
        self, query, kwargs: Union[None, dict] = None
    ) -> List[Document]:
        if kwargs is None:
            kwargs = {}
        relevant_docs = [d for i, d in enumerate(self.docs) if i in [0, 2]]
        return relevant_docs


class ChromaEmbedder(Embedder):
    def __init__(self, collection_name=None, persist_directory=None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._db = None
        self._client = None
        self._collection = None
        self.model = self._get_model()
        self.query_model = self._get_query_model()

    @property
    def collection_name(self):
        return self._collection_name

    def load_embeddings(self, in_embeddings: os.PathLike):
        pass

    @collection_name.setter
    def collection_name(self, collection_name):
        if collection_name is None:
            collection_name = "default_collection"
        new_collection_name = clean_collection_name_for_chroma(collection_name)
        LOG.warning(
            f"Changing collection name from {collection_name} to {new_collection_name} for chroma compatibility."
        )
        self._collection_name = new_collection_name

    def _get_model(self):
        model = embedding_functions.DefaultEmbeddingFunction()
        return model

    def _get_query_model(self):
        return self.model

    def create_embeddings(self, docs: List[Document], use_existing_collection=True):
        client = chromadb.Client() # Not persisting
        if use_existing_collection is True:
            collection = client.get_or_create_collection(
                name=self.collection_name, embedding_function=self.model)
        else:
            collection = client.create_collection(
                name=self.collection_name, embedding_function=self.model)
        if collection.count() == 0:
            LOG.info(f"Collection count: {collection.count()}")
            contents = [d.page_content for d in docs]
            metadatas = [d.metadata for d in docs]
            uuids = [str(uuid.uuid4()) for _ in docs]

            collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=uuids,
            )
        self._client = client
        self._collection = collection

    def get_relevant_docs(
        self, query, kwargs: Union[None, dict] = None, n_results=10,
    ) -> List[Document]:
        if kwargs is None:
            kwargs = {}
        # TODO: create the query embeddings using instructor then pass them in
        query_embeddings = self.query_model([query])
        results = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            **kwargs,
        )
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        langchain_docs = [Document(page_content=d, metadata=m) for d, m in zip(docs, metadatas)]
        return langchain_docs, results['distances'][0]
        # data_docs = [(d, s) for d, s in zip(langchain_docs, results['distances'][0])]
        # return data_docs
