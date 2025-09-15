import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import fitz
import pytesseract
from langchain_core.documents import Document
from langchain.text_splitter import TokenTextSplitter
from pdf2image import convert_from_bytes
from tqdm import tqdm
from unstructured.documents.elements import NarrativeText, FigureCaption, Title
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text_type import is_possible_narrative_text
from unstructured.partition.text_type import sentence_count

from paper_data_linking.utils import get_stage_message

LOG = logging.getLogger(__file__)


REFERNCES_TERMS = [
    "references",
    "bibliography",
    "works cited",
    "literature cited",
    "cited literature",
    "reference list",
    "sources",
    "citations",
    "bibliographic references",
]


def ocr_image(img):
    return pytesseract.image_to_string(img)


def is_junk_pdf(content, line_to_char_ratio_threshold=0.10, similarity_threshold=0.8):
    lines = content.strip().split("\n")
    num_lines = len(lines)
    num_chars = sum(len(line) for line in lines)

    if num_chars == 0:
        return True

    line_to_char_ratio = num_lines / num_chars
    if line_to_char_ratio > line_to_char_ratio_threshold:
        return True

    # Check similarity between lines using Counter
    line_counts = Counter(lines)
    most_common_line_count = line_counts.most_common(1)[0][1]
    similarity_ratio = most_common_line_count / num_lines

    if similarity_ratio > similarity_threshold:
        return True

    return False


def remove_references(text: str, reference_terms: list):
    # Convert all text to lower case for uniform comparison
    text_lower = text.lower()
    positions = []
    for term in reference_terms:
        # find the position for each term in the text
        pos = text_lower.rfind(term)
        if pos != -1:
            positions.append(pos)
    if positions:
        # Get the last occurrence among all terms
        last_pos = max(positions)
        # Slice the text to keep everything before the last occurrence
        text = text[:last_pos]
    return text


def is_junk(s):
    # Remove single and double quotations in string
    s = re.sub(r"['\"]", "", s)

    # Check for url-like patterns
    # url_pattern = re.compile(
    #     r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    # )
    # if re.search(url_pattern, s):
    #     return True

    # Check if the string has unusual spacing (a space after every character)
    unusual_spacing_pattern = re.compile(r"^((\w\s)+)$")
    if re.match(unusual_spacing_pattern, s):
        return True

    # Check if the string contains a high proportion of non-alphabetic characters
    non_alpha = len([c for c in s if not c.isalpha()])
    if non_alpha > len(s) / 2:
        return True

    # Check if the string matches a citation pattern
    # citation_pattern = re.compile(r"(\w+[\.,]\s*)+([\da-zA-Z]{2,}\.?.*|$)")
    # if re.match(citation_pattern, s.strip()):
    #     return True

    # If none of the above conditions were met, the string is not "junk"
    return False


def is_figure_caption(s):
    # Define the figure caption pattern
    figure_caption_pattern = re.compile(r"^Figure\s\d+\. .*$", re.DOTALL)

    # Check if the string matches the figure caption pattern
    return bool(re.match(figure_caption_pattern, s.strip()))


class Splitter(ABC):
    @abstractmethod
    def split(self, doc) -> List[Document]:
        pass


class FakeSplitter(Splitter):
    def split(self, content) -> List[Document]:
        splits = [
            Document(
                page_content="The sun is really neat",
                metadata={"page": 1, "x": 0, "y": 2, "width": 1, "height": 1},
            ),
            Document(
                page_content="The moon is pretty cool.",
                metadata={"page": 2, "x": 1, "y": 3, "width": 2, "height": 1},
            ),
        ]
        return splits


class UnstructuredSplitter(Splitter):
    def __init__(self):
        pass

    def _load_doc(self, content):
        if isinstance(content, str) or isinstance(content, Path):
            content = None
            doc = fitz.open(content)
        else:
            content = content.read()
            doc = fitz.open(stream=content, filetype="pdf")
        return doc

    @staticmethod
    def _split(doc) -> List[Document]:
        """
        Extracts the text and location information from the document.

        Returns:
            A list of MyDocument objects with text, location information, and langchain documents.
        """
        documents = []
        for page in tqdm(doc):
            blocks = page.get_text("blocks")
            for block in blocks:
                x, y, width, height, text, _, _ = block
                metadata = {
                    "page": page.number + 1,
                    "x": x,
                    "y": y,
                    "width": width - x,
                    "height": height - y,
                }
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
        return documents

    def _filter(self, elements):
        keep_categories = (Title, NarrativeText, FigureCaption)
        filtered_elements = []
        for e in elements:
            if isinstance(e, keep_categories) and not is_junk(e.text):
                filtered_elements.append(e)
        return filtered_elements

    def split(self, content) -> List[Document]:
        elements = partition_pdf(content, include_page_breaks=True)
        filtered_elements = self._filter(elements)
        tmp = "\n\n".join([f"{e.category}:{e.text}" for e in filtered_elements])
        with self._load_doc(content) as doc:  # make sure we close the fitz pdf
            all_splits = self._split(doc)
        filtered_splits = self._filter(all_splits)
        return filtered_splits


class PyMuPDFTokenSplitter(Splitter):
    def __init__(self, max_workers=None, update_progress=None):
        self.max_workers = max_workers
        self.update_progress = update_progress

    def _load_doc(self, content):
        if isinstance(content, Path) or isinstance(content, str):
            with open(content, "rb") as f:
                content_bytes = f.read()
            doc = fitz.open(stream=content_bytes, filetype="pdf")
        elif isinstance(content, bytes):
            doc = fitz.open(stream=content, filetype="pdf")
            content_bytes = content
        else:  # Assume content is a readable file-like object
            try:
                content_bytes = content.read()
                doc = fitz.open(stream=content_bytes, filetype="pdf")
            except Exception as e:
                raise TypeError(
                    "Content must be a Path, str, bytes, or a readable file-like object"
                ) from e
        return doc, content_bytes

    def _get_text_ocr(self, data, use_multiprocessing=False):
        if self.update_progress:
            m = get_stage_message(stage=-1)
            self.update_progress(m, "Reading PDF with OCR. This could take a while.")
        images = convert_from_bytes(data)

        if use_multiprocessing:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                txts = list(tqdm(executor.map(ocr_image, images), total=len(images)))
        else:
            txts = []
            total_images = len(images)
            for i, image in enumerate(tqdm(images)):
                txts.append(ocr_image(image))
                if self.update_progress:
                    m = get_stage_message(stage=-1)
                    self.update_progress(
                        m, f"OCR Progress: Page {i + 1}/{total_images}"
                    )
        txt = chr(12).join(txts)
        return txt

    def _get_text(self, content, coerce_ocr=False):
        doc, data = self._load_doc(content)
        if not coerce_ocr:
            txt = chr(12).join([p.get_text(sort=True) for p in doc])
            if not is_junk_pdf(txt):
                return txt, False

        LOG.info(
            "Failed to parsed with PyMuPDF, attempting to use pytesseract for OCR."
        )
        txt = self._get_text_ocr(data)
        if is_junk_pdf(txt):
            raise Exception("Failed to parse PDF with PyMuPDF and pytesseract OCR.")
        else:
            return txt, True

    def split(self, content) -> List[Document]:
        txt, ocr = self._get_text(content)
        all_splits = self._split(txt)
        filtered_splits = self._filter(all_splits)

        if len(filtered_splits) == 0:  # Try again with OCR
            txt, ocr = self._get_text(content, coerce_ocr=True)
            all_splits = self._split(txt)
            filtered_splits = self._filter(all_splits)

        return filtered_splits, ocr

    @staticmethod
    def _filter(docs: List[Document]) -> List[Document]:
        # maybe should filter out references. Find way to do this.
        filtered_docs = []
        for d in docs:
            text = d.page_content
            if (
                is_possible_narrative_text(text)
                and sentence_count(text) > 2
                and not is_junk(text)
            ):
                filtered_docs.append(d)
            else:
                continue
        return filtered_docs

    @staticmethod
    def _split(txt):
        txt = remove_references(txt, REFERNCES_TERMS)
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(txt)
        docs = []
        for i, t in enumerate(texts):
            docs.append(Document(page_content=t, metadata={"position": i}))
        return docs


class PyMuPDFSplitter(Splitter):
    def __init__(self):
        pass

    def _load_doc(self, content):
        if isinstance(content, str) or isinstance(content, Path):
            # content = None
            doc = fitz.open(content)
        else:
            content = content.read()
            doc = fitz.open(stream=content, filetype="pdf")
        return doc

    @staticmethod
    def _split(doc) -> List[Document]:
        """
        Extracts the text and location information from the document.

        Returns:
            A list of MyDocument objects with text, location information, and langchain documents.
        """
        documents = []
        for page in tqdm(doc):
            blocks = page.get_text("blocks")
            for block in blocks:
                x, y, width, height, text, _, _ = block
                metadata = {
                    "page": page.number + 1,
                    "x": x,
                    "y": y,
                    "width": width - x,
                    "height": height - y,
                }
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
        return documents

    @staticmethod
    def _filter(docs: List[Document]) -> List[Document]:
        # maybe should filter out references. Find way to do this.
        filtered_docs = []
        for d in docs:
            text = d.page_content
            if (
                is_possible_narrative_text(text)
                and sentence_count(text) > 2
                and not is_junk(text)
            ):
                filtered_docs.append(d)
            else:
                continue
        return filtered_docs

    def split(self, content) -> List[Document]:
        with self._load_doc(content) as doc:  # make sure we close the fitz pdf
            all_splits = self._split(doc)
        filtered_splits = self._filter(all_splits)
        return filtered_splits
