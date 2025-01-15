import io
from collections import Counter

import fitz
import pytesseract
from pdf2image import convert_from_bytes
from tqdm import tqdm

from paper_data_linking.log_config import logger


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


def get_text(content):
    """
    Get text of pdf from requests content.

    Args:
        content: content from requests or selenium download

    Returns:
        full text of the pdf
    """
    try:
        with io.BytesIO(content) as buffer:
            pdf_document = fitz.open(stream=buffer, filetype="pdf")
    except Exception as e:
        raise ValueError("File is probably not a pdf.") from e

    text = extract_text(pdf_document)
    if is_junk_pdf(text):
        logger.info("Got junk PDF. Attempting OCR instead.")
        text = pdf_to_text_with_ocr(content)
        if is_junk_pdf(text):
            raise ValueError("Could not get quality text")
    return text


def pdf_to_text_with_ocr(pdf_data):
    # Convert PDF bytes to a list of PIL images
    images = convert_from_bytes(pdf_data)
    # get the data which has the rectangles location in the pdf.

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

    Returns:
        full text of the pdf
    """
    try:
        with io.BytesIO(content) as buffer:
            pdf_document = fitz.open(stream=buffer, filetype="pdf")
    except Exception as e:
        raise ValueError("File is probably not a pdf.") from e

    data = extract_data(pdf_document)
    # test if junk
    text = '\n'.join([d[0][4] for d in data])
    if is_junk_pdf(text):
        logger.info("Got junk PDF. Attempting OCR instead.")
        data = pdf_to_data_with_ocr(content)
        text = '\n'.join([d[0][4] for d in data])
        if is_junk_pdf(text):
            raise ValueError("Could not get quality text")
    return data


def pdf_to_data_with_ocr(pdf_data):
    # Convert PDF bytes to a list of PIL images
    images = convert_from_bytes(pdf_data)
    # get the data which has the rectangles location in the pdf.

    # Extract text from the images using pytesseract
    data = []
    for img in tqdm(images):
        data.append(pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT))
    return data


def extract_data(pdf_document):
    data = []
    for page in pdf_document:
        data.append(page.get_text("words"))
    return data
