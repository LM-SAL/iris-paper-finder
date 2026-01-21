# IRIS Paper LLM

**This project was created by Buonomo, Anthony R.**

The goal of this repository is to be able to find, identify and extract data references in research papers which talk about IRIS.

An IRIS paper is defined as the following:

- Any paper which shows IRIS data
- Any paper which creates synthetic data of any IRIS passband

The `paper-data-linking` library contains code for ingesting and downloading PDFs.
There is also a web application which will allow one to upload PDFs and have them analyzed by a LLM.

## **Installation**

To run the library, you can pip install the library into your virtual environment with the "scrape" extras.

```bash
pip install -e ".[scrape]"
```

### Getting Papers

This part of the library is managed via the Makefile.

```bash
make help
```

will provide you the targets and the order of the commands for normal use.

The queries to the ADS are fixed and one should modify the Makefile.
Currently it will fetch the IRIS ADS library and do a query for all papers which have cited the IRIS instrument paper.

Typically when downloading the papers, you will hit bot procetions.
Within `src/paper_data_linking/data/headers.py` is the code which creates the request headers.
This might need updating or the bot protection is too advanced to bypass which means you will need to manually download those papers.

When you have them all downloaded, you can precede with the rest of the readme.

## Web Application

To run the API, you will need to set up Docker, Docker Compose, and a few configuration files.

### **1. Prerequisites**

Make sure you have the following installed on your system:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)

### **2. Configure Environment Variables**

1. Copy the example `.env` file:

   ```bash
   cp .env_example .env
   ```

2. Open the `.env` file and fill in the required environment variables.

### **3. Download ONNX Model**

1. Run the following make command:

   ```bash
   make onnx
   ```

### **4. Build Docker Images**

Navigate to the root of the repository and build the Docker images:

```bash
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose build
```

This process may take a few minutes.

### **5. Start the Services**

Run the application:

```bash
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose up
```

(The reason the env variables are still used here, is that its easier to just replace build with up in the commandline history)

The services may take a minute or two to start up.

### **6. Access the Application**

1. Open your browser and navigate to:

   ```bash
   http://localhost:80
   ```

2. If you see an NGINX 500 error, wait a moment for the `web` service to fully initialize.

### **Notes**

- For troubleshooting, ensure Docker is running and check logs using:

  ```bash
  docker-compose logs
  ```

- If you encounter any issues, refer to the project's documentation or contact support.

## Usage of the Web App

How to use the Paper Analyzer:

1. **Select a Classifier**
   - Use the dropdown menu to select the appropriate classifier for your analysis
   - Different classifiers are optimized for different types of papers and analysis goals
   - For instance, you might select `Interface Region Imaging Spectrograph`.

2. **Upload Your PDF**
   - Click "Choose File" to select your PDF document
   - The system accepts standard PDF files
   - There is an example PDF file available in the `scripts` dir of this repo.

3. **Start Analysis**
   - Click the "Analyze" button to begin processing.
   - The system will show progress indicators for:
     - Parsing: Initial PDF text extraction
     - Embedding: Creating vector embeddings for the text chunks to enable text chunk relevance ranking
     - Analyzing: Running the selected classifier
   - Analysis typically takes a few seconds to a minute depending on the PDF size.

4. **View Results**
   - Results appear in three panels:
     - Text from PDF: Shows the extracted text with highlights
     - LLM Analysis: Detailed analysis of the content
     - JSON Output: Structured data output
   - Results can be copied or downloaded for further use

### API Endpoints

For programmatic access, the following endpoints are available:

- `GET /api/get_configs`: Retrieve available classifier configurations
- `POST /api/upload_pdf`: Upload a PDF file for analysis
- `GET /api/task/{task_id}`: Check status of an analysis task
- `POST /api/highlight_pdf`: Process a PDF with highlights

Note: The API has a rate limit of 20 requests per minute.

You can see an example of how to programmatically use the API endpoints to do a prediction in the file [scripts/query_api.py](scripts/query_api.py).
Note that this script requires the `requests` and `python-dotenv` libraries in order to run.

This is how this is meant to be run.
Using the Makefile, you can do a ADS query, get the PDFs and then use the script above to do the LLM processing.

### Tips

- Ensure your PDF is text-searchable for best results
- Large files may take longer to process
- Check the classifier descriptions to choose the most appropriate one for your paper

## Analyzer Configuration Guide

The analyzer uses a YAML configuration file to define how it processes scientific papers.
This guide explains each component and how they work together.

## Processing Pipeline

1. **Initial Filtering**: Uses fuzzy string matching to quickly identify potentially relevant content
2. **Embedding & Retrieval**: Creates embeddings of the text and finds relevant sections
3. **LLM Analysis**: Analyzes the relevant sections to extract detailed information

## Configuration File Structure

### Top-Level Fields

```yaml
name: "Interface Region Imaging Spectrograph" # Name of the instrument/mission
classifier: # Main configuration block
  # Classifier settings detailed below
model_kwargs: # LLM model settings
  model_name: "gpt-4"
  temperature: 0
embedder_kwargs: # Embedding settings
  where:
    passed_iris_heuristic: 1
```

### Classifier Configuration

```yaml
classifier:
  name: "IRIS" # Identifier for this classifier
  query: "Does this paper use data from the IRIS spacecraft or its instruments?" # Query for relevant content
  system_message: | # Context for the LLM
    # Background information about the mission/instrument
  human_message: | # Instructions for analysis
    # Specific questions and format requirements
  answer_divider: "Classification:" # Marker to extract classification
  json_divider: "IRIS Aspects Used:" # Marker to extract structured data
  answer_key: "IRIS" # Key for storing classification
  json_key: "aspects" # Key for storing structured data
  filter_terms: # Terms for initial filtering
    - "IRIS"
    - "Interface Region Imaging Spectrograph"
    # ... more terms
  filter_threshold: 80 # Fuzzy matching threshold (0-100)
```

### Metadata Mapping

```yaml
label_metadata_map:
  IRIS Telescope: # Component name
    link: "https://iris.lmsal.com/" # Reference link
    detail: "High-resolution solar observation instrument" # Description
```

## How It Works

### 1. Initial Filtering

The `is_soho_related()` function performs initial filtering using:

- `filter_terms`: List of relevant terms to look for
- `filter_threshold`: Minimum fuzzy match score (0-100) to consider a match
- Uses sentence-level matching with spaCy
- Returns True if ANY term matches above threshold

Passages which do not contain any of these terms (or close matches) will NOT be analyzed by the LLM.
Moreover, if not a single passage contains on of these terms (or a close match), no LLM analysis will even occur and we will assume the paper is not relevant.
Therefore, one should be reasonably certain that if a paper does not contain any of these terms, the paper is not relevant.

### 2. Embedding & Retrieval

Uses the `embedder_kwargs` to:

- Create embeddings of document sections
- Find sections relevant to the `query`
- Filters based on the initial filtering results using the `where` clause

### 3. LLM Analysis

The LLM:

1. Receives context via `system_message`
2. Processes relevant sections using `human_message` instructions
3. Returns structured output based on the message templates
4. Results are extracted using `answer_divider` and `json_divider`

## Customization Guide

To modify the analyzer for a new instrument:

1. Update the top-level `name` and classifier `name`
2. Modify the `query` for your instrument
3. Update `system_message` with relevant background
4. Adjust `human_message` for desired analysis
5. Add appropriate `filter_terms`
6. Define instrument components in `label_metadata_map`

Key considerations:

- Filter terms should be distinctive to minimize false positives
- System message should provide comprehensive context
- Human message should clearly specify output format
- Filter threshold may need adjustment based on term uniqueness

## Configuration Management

The application uses configuration files to define how it processes scientific papers.
These configurations are built directly into the Docker image and are available to all services using that image.

### Configuration Structure

- `src/paper_data_linking/web_app/config/`: Contains all configuration YAML files
  - Analysis rules for different instruments
  - LLM prompts and parameters
  - Filter terms and thresholds
  - Metadata mappings

### Managing Configurations

1. **Making Changes**:
   - Edit YAML files in the `src/paper_data_linking/web_app/config/` directory
   - Rebuild the Docker image
   - Restart services to use the new image
