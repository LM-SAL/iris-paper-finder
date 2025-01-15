# paper-data-linking

Identify and extract data references in heliophysics research papers.

## Description
Tools for information extraction on heliophysics papers. The `paper-data-linking` library contains code for ingesting and analyzing pdfs in order to enrich their metadata.

## **Installation**

To run the API, you’ll need to set up Docker, Docker Compose, and a few configuration files. Follow these steps:

---

### **1. Prerequisites**
Make sure you have the following installed on your system:
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)

---

### **2. Configure Environment Variables**
1. Copy the example `.env` file:
   ```bash
   cp .env_example .env
   ```
2. Open the `.env` file and fill in the required environment variables under the `App variables` section. Don't worry about `App credentials` just yet.

---

### **3. Set Up Authentication**
1. Navigate to the `nginx` directory:
   ```bash
   cd nginx
   ```
2. Create a password file for the reverse proxy:
   ```bash
   htpasswd -c .htpasswd <USERNAME>
   ```
   Replace `<USERNAME>` with your desired username, then enter your desired password when prompted.
3. Return to the root directory:
   ```bash
   cd ..
   ```
4. Now fill in the `App Credentials` section from the `.env` file with these chosen values.

---

### **4. Download ONNX Model**
1. Create a `models` directory if it doesn’t exist:
   ```bash
   mkdir -p models
   ```
2. Navigate to the `models` directory:
   ```bash
   cd models
   ```
3. Download the `all-MiniLM-L6-v2` ONNX model:
   ```bash
   wget https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
   ```
4. Extract the model:
   ```bash
   tar -xzvf onnx.tar.gz
   ```
5. Return to the root directory:
   ```bash
   cd ..
   ```

---

### **5. Build Docker Images**
Navigate to the root of the repository and build the Docker images:
```bash
docker compose build
```
This process may take a few minutes.

---

### **6. Start the Services**
Run the application:
- For **production mode**:
  ```bash
  docker compose -f docker-compose.yaml up
  ```
- For **development mode**:
  ```bash
  docker compose up
  ```
This will also use the `docker-compose.override.yaml` file. The services may take a minute or two to start up.

---

### **7. Access the Application**
1. Open your browser and navigate to:
   ```
   http://localhost:80
   ```
2. If you see an NGINX 500 error, wait a moment for the `web` service to fully initialize.
3. Log in with:
   - **Username**: The one you created in step 3.
   - **Password**: The password you entered in step 3.

---

### **Notes**
- For troubleshooting, ensure Docker is running and check logs using:
  ```bash
  docker-compose logs
  ```
- If you encounter any issues, refer to the project’s documentation or contact support.


## Usage

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

You can see an example of how to programmatically use the API endpoints to do a prediction in the file [scripts/query_api.py](scripts/query_api.py). Note that this script requires the `requests` and `python-dotenv` libraries in order to run.

### Tips

- Ensure your PDF is text-searchable for best results
- Large files may take longer to process
- Check the classifier descriptions to choose the most appropriate one for your paper

# Analyzer Configuration Guide

The analyzer uses a YAML configuration file to define how it processes scientific papers. This guide explains each component and how they work together.

## Processing Pipeline

1. **Initial Filtering**: Uses fuzzy string matching to quickly identify potentially relevant content
2. **Embedding & Retrieval**: Creates embeddings of the text and finds relevant sections 
3. **LLM Analysis**: Analyzes the relevant sections to extract detailed information

## Configuration File Structure

### Top-Level Fields

```yaml
name: "Interface Region Imaging Spectrograph"  # Name of the instrument/mission
classifier:  # Main configuration block
  # Classifier settings detailed below
model_kwargs:  # LLM model settings 
  model_name: "gpt-4"
  temperature: 0
embedder_kwargs:  # Embedding settings
  where:
    passed_iris_heuristic: 1
```

### Classifier Configuration

```yaml
classifier:
  name: "IRIS"  # Identifier for this classifier
  query: "Does this paper use data from the IRIS spacecraft or its instruments?"  # Query for relevant content
  system_message: |  # Context for the LLM
    # Background information about the mission/instrument
  human_message: |  # Instructions for analysis
    # Specific questions and format requirements
  answer_divider: "Classification:"  # Marker to extract classification
  json_divider: "IRIS Aspects Used:"  # Marker to extract structured data
  answer_key: "IRIS"  # Key for storing classification
  json_key: "aspects"  # Key for storing structured data
  filter_terms:  # Terms for initial filtering
    - "IRIS"
    - "Interface Region Imaging Spectrograph"
    # ... more terms
  filter_threshold: 80  # Fuzzy matching threshold (0-100)
```

### Metadata Mapping

```yaml
label_metadata_map:
  IRIS Telescope:  # Component name
    link: "https://iris.lmsal.com/"  # Reference link
    detail: "High-resolution solar observation instrument"  # Description
```

## How It Works

### 1. Initial Filtering

The `is_soho_related()` function performs initial filtering using:
- `filter_terms`: List of relevant terms to look for
- `filter_threshold`: Minimum fuzzy match score (0-100) to consider a match
- Uses sentence-level matching with spaCy
- Returns True if ANY term matches above threshold

Passages which do not contain any of these terms (or close matches) will NOT be analyzed by the LLM. Moreover, if not a single passage contains on of these terms (or a close match), no LLM analysis will even occur and we will assume the paper is not relevant. Therefore, one should be reasonably certain that if a paper does not contain any of these terms, the paper is not relevant.

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

The application uses configuration files to define how it processes scientific papers. These configurations are built directly into the Docker image and are available to all services using that image.

### Configuration Structure

- `config/`: Contains all configuration YAML files
  - Analysis rules for different instruments
  - LLM prompts and parameters
  - Filter terms and thresholds
  - Metadata mappings

### Docker Setup

The configuration files are built directly into the Docker image during build time:
```dockerfile
# In Dockerfile
COPY config /code/config/
```

Since both web and celery services use the same Dockerfile, they automatically share identical configurations.

The entrypoint script verifies configurations are present:
```bash
#!/bin/bash

# Check if the config directory has yaml files
if [ ! -d "/code/config" ] || [ -z "$(ls -A /code/config/*.yaml 2>/dev/null)" ]; then
    echo "Error: No configuration files found in /code/config/"
    exit 1
fi

# Execute the provided command
exec "$@"
```

### Managing Configurations

1. **Making Changes**:
   - Edit YAML files in the `config/` directory
   - Rebuild the Docker image
   - Restart services to use the new image

2. **Development/Production**:
   ```bash
   # Build image with updated configs
   docker compose build
   
   # Start services (both web and celery will use same configs)
   docker compose up
   ```

This approach ensures:
- Configuration changes are version-controlled
- All services use identical configurations
- No runtime configuration management needed
- Simple, reliable deployment process