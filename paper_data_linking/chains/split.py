import re

from langchain.llms import LlamaCpp
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings, FakeEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
from langchain.llms import OpenAI, OpenAIChat
from langchain.indexes import VectorstoreIndexCreator

from paper_data_linking.enums import ModelPaths
from paper_data_linking.log_config import logger
from paper_data_linking.enums import Locations
from paper_data_linking.settings import OPENAI_API_KEY


def clean_newlines(s):
    # Replace single newlines with a placeholder
    s = re.sub(r'(?<!\n)\n(?!\n)', 'PLACEHOLDER', s)

    # Replace double newlines, but not when there are two double-newlines with a space in the middle
    s = re.sub(r'(?<!\n\n \n)\n\n(?! \n\n)', 'PLACEHOLDER', s)

    # Remove the placeholder and return the result
    return s.replace('PLACEHOLDER', '')


def clean_double_column():
    # Double-column text has shortened lines with many hyphenated words at the line breaks.
    # Paragraph breaks are not very easy to tell since they are determined by noticing that a
    # line ends early. There is often no double-space.
    pass


def has_extraneous_newlines(text, threshold=0.2):
    """
    Heuristically try to determine if a given text has extraneous newlines or not based on the ratio of single newline characters
    to the total number of newline characters.

    Args:
        text (str): The input text to be analyzed.
        threshold (float): The threshold for the ratio of single newline characters to total newline characters.
        If the ratio is greater than the threshold, the text is considered weird. Default is 0.2.

    Returns:
        bool: True if the text may have extraneous newlines, False otherwise.
    """
    single_newlines = len(re.findall(r'(?<!\n)\n(?!\n)', text))
    total_newlines = len(re.findall(r'\n', text))

    if total_newlines == 0:
        return False

    single_newline_ratio = single_newlines / total_newlines
    return single_newline_ratio < threshold, single_newline_ratio


def outer_split(bibcode):
    infile = Locations.TEXTS.value / bibcode
    with open(infile, "r") as f0:
        txt = f0.read().strip()
    out_vectorstore = Locations.VECTORSTORES_DIR.value / f"{bibcode}.db"
    make_vectorstore(txt, out_vectorstore)


def make_vectorstore(txt, out_vectorstore):
    # Text cleaning
    weird, _ = has_extraneous_newlines(txt)
    if weird:
        txt = clean_newlines(txt)

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=20,
    )
    docs = text_splitter.create_documents([txt])
    print(len(docs))

    # Embedding
    embeddings = OpenAIEmbeddings()
    # perhaps should have a "collection" for a given document and have all
    # doc chunks in same db. That way we can later do queries across the whole db.
    db = Chroma.from_documents(docs, embeddings, persist_directory=str(out_vectorstore))
    return db


def load_vectorstore(in_vectorstore):
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=str(in_vectorstore), embedding_function=embeddings)

    docs = db.similarity_search("Data sources", k=10) # for some reason, k is halved.
    # list of docs, where doc text is available at docs.page_content
    return docs

    retriever = db.as_retriever()
    # llm = LlamaCpp(model_path=str(ModelPaths.ALPACA_7B.value), temperature=0, max_tokens=64)
    # llm = OpenAI(model="text-ada-001", temperature=0)
    llm = OpenAIChat(model="gpt-3.5-turbo", temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="refine", return_intermediate_steps=True)
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)
    query = "Concise list of data used?"
    ans = qa.run(query)
    # print(ans)


if __name__ == "__main__":
    # outer_split()
    pass