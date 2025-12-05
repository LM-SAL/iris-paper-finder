"""
This whole file should live in paper-data-linking, not the api dir.

This should just have the API methods.
"""

import os
import json
from uuid import uuid4

import openai
import yaml

from paper_data_linking.process.analyzer import get_analyzer
from paper_data_linking.utils import get_time
from paper_data_linking.web_app.celery_app import app
from paper_data_linking.web_app.settings import CONFIG_DIR, OPENAI_API_KEY

openai.openai_api_key = OPENAI_API_KEY


def get_config_files():
    config_dir = CONFIG_DIR
    config_list = []
    for f in os.listdir(config_dir):
        if os.path.isfile(os.path.join(config_dir, f)):
            with open(os.path.join(config_dir, f)) as file:
                content = yaml.safe_load(file)
                name = content.get("name", "Unnamed")
                config_list.append({"filename": f, "name": name})
    return config_list


def get_location_dct(doc, intensity=1, color=(255, 255, 0)):
    return {
        "page": doc.metadata["page"],
        "rect": {
            "x": doc.metadata["x"],
            "y": doc.metadata["y"],
            "width": doc.metadata["width"],
            "height": doc.metadata["height"],
        },
        "intensity": intensity,
        "color": color,
    }


def get_rectangles_from_docs(docs, color=(255, 255, 0)):
    rects = []
    for dd, score in docs:
        rect = get_location_dct(dd, intensity=score, color=color)
        rects.append(rect)
    return rects


def prepare_records_for_frontend(records):
    all_rects = []
    full_analysis = ""
    all_data = []
    for r in records:
        analysis = "" if r.analysis is None else r.analysis
        full_analysis = full_analysis + "\n\n---\n\n" + analysis
        all_data.append(json.loads(r.json()))
    return all_rects, full_analysis, all_data


@app.task(bind=True)
def get_docs_for_frontend(self, pdf_file, filename, config_file, analyzer=None):
    def update_progress(percentage, message):
        if self.request.id:
            self.update_state(
                state="PROGRESS",
                meta={"current": percentage, "total": 100, "status": message},
            )

    config_loc = os.path.join(CONFIG_DIR, config_file)
    text, rect_locations, analysis, json_data = get_docs(
        pdf_file, filename, [config_loc], update_progress=update_progress, analyzer=analyzer
    )
    return {
        "highlights": rect_locations,
        "analysis": analysis,
        "data": json_data,
        "text": text,
    }


def get_full_text(docs: list, relevant_docs=None) -> str:
    if relevant_docs is None:
        relevant_docs = []
    relevant_positions = [d.metadata["position"] for d in relevant_docs]
    full_text = ""
    for d in docs:
        pos = d.metadata["position"]
        content = d.page_content
        if pos in relevant_positions:
            content = f'<span style="background-color: #FFFF00">{content}</span>'
        txt = f"<h2>Excerpt {pos + 1}</h2>\n\n<blockquote><p>{content}</p></blockquote>"
        full_text += txt
    return full_text


def get_docs(pdf_file, filename, config_files, analyzer=None, update_progress=None):
    if analyzer is None:
        analyzer = get_analyzer(config_files, update_progress=update_progress)
    else:
        analyzer.splitter.update_progress = update_progress

    analysis_results = analyzer.process(pdf_file, update_progress=update_progress)

    docs = analysis_results.docs
    records: list = analysis_results.records

    # Let's gather all the relevant docs from the records
    # This is highlighting relevant docs for all plugins in one spot
    all_relevant_docs = []
    for record in records:
        if record.relevant_indices is None:
            all_relevant_docs.extend([])
        else:
            plugin_rel_docs = [docs[i] for i in record.relevant_indices]
            all_relevant_docs.extend(plugin_rel_docs)

    # Using the get_full_text function to join all docs and highlight relevant ones
    text = get_full_text(docs, all_relevant_docs)

    try:
        rects, analysis, data = prepare_records_for_frontend(records)
        message = "success"
    except Exception:
        rects = []
        analysis = ""
        data = []
        message = "failure"

    return_data = {
        "uuid": str(uuid4()),
        "timestamp": get_time(),
        "message": message,
        "filename": filename,
        "results": data,
    }

    return text, rects, analysis, return_data
