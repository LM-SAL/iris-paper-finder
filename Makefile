IRIS_LIB_ID=30bDOCvOTJiAgacWhJxkmA

# Services
bibcode_service=src/paper_data_linking/data/bibcode_service.py
metadata_script=src/paper_data_linking/data/metadata_service.py
url_service=src/paper_data_linking/data/url_service.py
pdf_download_service=src/paper_data_linking/data/pdf_download_service.py

# Bibcode output paths
iris_search_bibcodes_path=data/bibcodes/iris_search_bibcodes.txt
iris_library_bibcodes_path=data/bibcodes/iris_library_bibcodes.txt

# Metadata output paths
iris_search_metadata_path=data/metadata/iris_search_metadata.jsonl
iris_library_metadata_path=data/metadata/iris_library_metadata.jsonl

# Transformed links output paths
iris_search_transformed_links_path=data/links/iris_search_transformed_links.jsonl
iris_library_transformed_links_path=data/links/iris_library_transformed_links.jsonl

# pdf output dirs sentinels
pdfs_dir=data/pdfs
pdf_output_dir_iris_search=$(pdfs_dir)/iris_search
pdf_output_dir_iris_library=$(pdfs_dir)/iris_library

iris_search_pdfs=$(pdf_output_dir_iris_search)/.done
iris_library_pdfs=$(pdf_output_dir_iris_library)/.done

# processed data output for evaluation
iris_search_processed_data_output=data/processed/iris_search_processed_data.jsonl
iris_library_processed_data_output=data/processed/iris_library_processed_data.jsonl

# Bibcode collection targets
$(iris_library_bibcodes_path):
	python $(bibcode_service) --library_id $(IRIS_LIB_ID) --output $(iris_library_bibcodes_path)

# This is for results which cite the main IRIS instrument paper or contains the full pharse: Interface Region Imaging Spectrograph
# You can add pubdate:[2025-05 TO 2026-01] if you want to limit by publication date
$(iris_search_bibcodes_path):
	python $(bibcode_service) --query '=full:"Interface Region Imaging Spectrograph" OR citations(bibcode:2014SoPh..289.2733D) + property:refereed + doctype:"Article" + pubdate:[2025-01 TO 2025-12]' --output $@

# Metadata collection targets with dependencies
$(iris_search_metadata_path): $(iris_search_bibcodes_path)
	python $(metadata_script) $< $@

$(iris_library_metadata_path): $(iris_library_bibcodes_path)
	python $(metadata_script) $< $@

# URL Transformation targets with dependencies
$(iris_search_transformed_links_path): $(iris_search_metadata_path)
	python $(url_service) $< $@

$(iris_library_transformed_links_path): $(iris_library_metadata_path)
	python $(url_service) $< $@

# PDF download targets with dependencies
$(iris_search_pdfs): $(iris_search_transformed_links_path)
	python $(pdf_download_service) --input $< --output-dir $(pdf_output_dir_iris_search)
	touch $@

$(iris_library_pdfs): $(iris_library_transformed_links_path)
	python $(pdf_download_service) --input $< --output-dir $(pdf_output_dir_iris_library)
	touch $@

build:  ## Build the Docker image for the web app
	docker compose build

bibcodes: ## Collect bibcodes from IRIS library and search results
	@echo "Collecting bibcodes..."
	$(MAKE) $(iris_search_bibcodes_path) $(iris_library_bibcodes_path)

metadata: ## Download metadata for collected bibcodes
	$(MAKE) $(iris_search_metadata_path) $(iris_library_metadata_path)

transform_links: ## Transform URLs for PDF access
	$(MAKE) $(iris_search_transformed_links_path) $(iris_library_transformed_links_path)

pdfs: ## Download PDFs from transformed links
	$(MAKE) $(iris_search_pdfs) $(iris_library_pdfs)

# Download ONNX models for embeddings
ONNX_URL=https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
onnx: ## Get onnx models
	cd models/ && \
	wget $(ONNX_URL) && \
	tar -xzvf onnx.tar.gz

clean: ## Clean generated data files
	rm -rf data/bibcodes data/metadata data/links data/pdfs

.PHONY: bibcodes metadata transform_links pdfs onnx clean
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo ''
	@echo 'IRIS Paper LLM Project'
	@echo ''
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?# "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ''
	@echo 'Ensure you have Python and Docker installed, and the necessary Python environment set up before running the pipeline.'
	@echo ''
	@echo 'You can also update the query in the Makefile to change the search criteria for collecting bibcodes.'
	@echo ''
	@echo 'Pipeline order:'
	@echo '  1. bibcodes     - Collect paper references'
	@echo '  2. metadata     - Download paper metadata'
	@echo '  3. transform_links - Transform URLs for PDF access'
	@echo '  4. pdfs         - Download PDFs'
	@echo ''
	@echo ' From here, you can run the processing step from the Python script in the scripts folder'
