import paper_data_linking.process.splitters as sp
from pathlib import Path
import fitz

from chains.split import clean_newlines, has_extraneous_newlines
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import TokenTextSplitter

# in_pdf = Path(__file__).parent / "data" / "soho_several_instruments.pdf"
# in_pdf = Path(__file__).parent / "data" / "agu.pdf"
in_pdf = Path(__file__).parent / "data" / "line_breaks.pdf"


def test__unstructured_split():
    splitter = sp.UnstructuredSplitter()
    splits = splitter.split(in_pdf)
    assert False


def test_clean_txt():
    doc = fitz.open(in_pdf)
    txt = chr(12).join([p.get_text(sort=True) for p in doc])
    weird, _ = has_extraneous_newlines(txt)
    if weird:
        txt = clean_newlines(txt)
    text_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=20)
    texts = text_splitter.split_text(txt)
    tmp = "\n\n----\n\n".join(texts)
    print(texts)


def test__pymupdf_split():
    splitter = sp.PyMuPDFSplitter()
    splits = splitter.split(in_pdf)
    assert False
