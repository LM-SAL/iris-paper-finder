from pathlib import Path

import pytest
from unstructured.partition.text_type import sentence_count

import paper_data_linking.chains.split as sp
from paper_data_linking.data.parsers import is_junk_pdf
from unstructured.documents.elements import NarrativeText, FigureCaption

testdata = [
    # ("2022Atmos..13.1781G", True),
    # ("2023ApJ...942...19S", True),
    # ("2023ApJ...943...70S", True),
    # ("2022A&A...662L...5M", False),
    ("2022yCat..36700051R", True)
    # ^ Does not require newline cleaning
]

testpdfs = [
    # ("2211.06775.pdf"),
    ("2211.11054.pdf"), # some sentences missing when not using fast
    # ("1995SoPh__162__357B.pdf"),
]


@pytest.mark.parametrize("filename,needs_cleaning", testdata)
def test_is_weird(filename, needs_cleaning):
    infile = Path(__file__).parent / filename
    with open(infile, "r") as f0:
        txt = f0.read().strip()
    weird, _ = sp.has_extraneous_newlines(txt)
    assert weird == needs_cleaning
    # assert isinstance(txt, str)


@pytest.mark.parametrize("filename,needs_cleaning", testdata)
def test_make_vectorstore(filename, needs_cleaning):
    infile = Path(__file__).parent / filename
    with open(infile, "r") as f0:
        txt = f0.read().strip()
    out_vectorstore = Path(__file__).parent / f"{filename}.db"
    sp.make_vectorstore(txt, out_vectorstore)
    assert out_vectorstore.exists()


def test_load_vectorstore():
    filename = "2022yCat..36700051R"
    in_vectorstore = Path(__file__).parent / f"{filename}.db"
    docs = sp.load_vectorstore(in_vectorstore)
    assert False


@pytest.mark.parametrize("filename", testpdfs)
def test_unstructured(filename):
    from unstructured.partition.auto import partition_pdf

    infile = Path(__file__).parent.parent / "data" / filename

    elements = partition_pdf(filename=infile, strategy="fast")
    # elements = partition_pdf(filename=infile, strategy="fast")
    txt = "\n\n".join([str(el) for el in elements])
    if is_junk_pdf(txt):
        elements = partition_pdf(filename=infile)

    useful_elements = []
    for element in elements:
        if isinstance(element, (NarrativeText, FigureCaption)) and sentence_count(element.text) > 2:
            useful_elements.append(element)

    txt = "\n\n".join([str(el) for el in useful_elements])
    print(txt)