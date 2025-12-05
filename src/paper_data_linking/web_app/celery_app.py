from celery import Celery

from paper_data_linking.web_app.settings import CELERY_ALWAYS_EAGER, REDIS_HOST

CELERY_REDIS_URL = f"redis://{REDIS_HOST}:6379"

app = Celery(
    "tasks",
    broker=CELERY_REDIS_URL,
    backend=CELERY_REDIS_URL,
    include=["paper_data_linking.web_app.api.pdf_processing"],
)
app.conf.update(
    {
        "task_always_eager": CELERY_ALWAYS_EAGER,
        "task_store_eager_result": True,  # if you want to store the result even in eager mode
    }
)
