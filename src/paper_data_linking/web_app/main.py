import logging
from typing import Annotated

import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from paper_data_linking.web_app.api.pdf_processing import get_config_files, get_docs_for_frontend
from paper_data_linking.web_app.settings import STATIC_DIR, TEMPLATES_DIR

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
router = APIRouter()
app.include_router(router)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

load_dotenv(find_dotenv())


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/get_configs")
async def get_configs():
    return {"configs": get_config_files()}


@app.post("/api/upload_pdf")
async def upload_pdf(file: Annotated[UploadFile, File()] = ...):  # NOQA: ARG001 - Used by the JS frontend
    return {"status": "success"}


@app.get("/api/task/{task_id}")
async def task_status(task_id: str):
    task = get_docs_for_frontend.AsyncResult(task_id)
    return {"task_status": task.status, "task_result": task.result}


@app.post("/api/highlight_pdf")
@limiter.limit("20/minute")
async def highlight_pdf(
    request: Request,  # NOQA: ARG001 - Actually required
    file: Annotated[UploadFile, File()] = ...,
    config_file: Annotated[str, Form()] = ...,
):
    file_bytes = file.file.read()
    # Asynchronous execution via Celery
    task = get_docs_for_frontend.delay(file_bytes, file.filename, config_file)
    return {"task_id": str(task.id), "status": task.status}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # NOQA: S104
