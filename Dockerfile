FROM python:3.10.13-slim

ARG YOUR_ENV

ENV YOUR_ENV=${YOUR_ENV} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.4.2


# Install necessary packages
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install gcc g++ python3-dev poppler-utils tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# System deps:
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi --no-root

RUN pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

RUN python -m nltk.downloader punkt && \
    python -m nltk.downloader averaged_perceptron_tagger

# create the app user
RUN adduser --system --group app

ADD models/onnx /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx
ADD models/onnx.tar.gz /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz

# Creating folders, and files for a project:
ADD app app
ADD paper_data_linking paper_data_linking
ENV PYTHONPATH "${PYTHONPATH}:/code/"

RUN chown -R app:app /code/
#USER app

WORKDIR app/

# Copy the entrypoint script to the container
COPY config /code/config/
COPY entrypoint.sh /code/entrypoint.sh
RUN chmod +x /code/entrypoint.sh

# Use the entrypoint script as the default entrypoint
ENTRYPOINT ["/code/entrypoint.sh"]
