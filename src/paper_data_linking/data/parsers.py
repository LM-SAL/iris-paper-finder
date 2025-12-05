import io
from collections import Counter

import fitz
import pytesseract
from pdf2image import convert_from_bytes
from tqdm import tqdm

from paper_data_linking import logger


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
    return similarity_ratio > similarity_threshold


def get_text(content):
    """
    Get text of pdf from requests content.

    Args:
        content: content from requests or selenium download

    Returns
    -------
        full text of the pdf
    """
    try:
        with io.BytesIO(content) as buffer:
            pdf_document = fitz.open(stream=buffer, filetype="pdf")
    except Exception as e:
        msg = "File is probably not a pdf."
        raise ValueError(msg) from e

    text = extract_text(pdf_document)
    if is_junk_pdf(text):
        logger.info("Got junk PDF. Attempting OCR instead.")
        text = pdf_to_text_with_ocr(content)
        if is_junk_pdf(text):
            msg = "Could not get quality text"
            raise ValueError(msg)
    return text


def pdf_to_text_with_ocr(pdf_data):
    # Convert PDF bytes to a list of PIL images
    images = convert_from_bytes(pdf_data)
    # Get the data which has the rectangles location in the pdf.
    # Extract text from the images using pytesseract
    text = ""
    for img in tqdm(images):
        text += pytesseract.image_to_string(img)
    return text


def extract_text(pdf_document):
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text


def get_data(content):
    """
    Get text of pdf from requests content.

    Args:
        content: content from requests or selenium download

    Returns
    -------
        full text of the pdf
    """
    try:
        with io.BytesIO(content) as buffer:
            pdf_document = fitz.open(stream=buffer, filetype="pdf")
    except Exception as e:
        msg = "File is probably not a pdf."
        raise ValueError(msg) from e
    data = extract_data(pdf_document)
    # test if junk
    text = "\n".join([d[0][4] for d in data])
    if is_junk_pdf(text):
        logger.info("Got junk PDF. Attempting OCR instead.")
        data = pdf_to_data_with_ocr(content)
        text = "\n".join([d[0][4] for d in data])
        if is_junk_pdf(text):
            msg = "Could not get quality text"
            raise ValueError(msg)
    return data


def pdf_to_data_with_ocr(pdf_data):
    # Convert PDF bytes to a list of PIL images
    images = convert_from_bytes(pdf_data)
    # Get the data which has the rectangles location in the pdf.
    # Extract text from the images using pytesseract
    return [pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT) for img in tqdm(images)]


def extract_data(pdf_document):
    return [page.get_text("words") for page in pdf_document]
