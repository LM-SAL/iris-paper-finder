#################################################################################
# COMMANDS                                                                 #
#################################################################################

CHROME_PATH=/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome
CHROMEDRIVER_URL=https://chromedriver.storage.googleapis.com
OS_PATH=chromedriver_mac64.zip # will change if linux...
PYTHONPATH := $(PYTHONPATH):$(CURDIR)

WIND_LIB_ID=DdelsbGYT5a8sM7CLujv_g

DATA_DIR=data
METADATA_RECORDS=$(DATA_DIR)/metadata_records.jsonl
BIBCODE_URLS=$(DATA_DIR)/bibcode_urls.jsonl
PUBDATE_CUTOFF=2022-11-01

# Might need to run
# python -m spacy download en_core_web_sm
# to download spacy models.

#py=poetry run python
py=python

# Need to install java8 for tika to work.
install:
	poetry run python -m spacy download en_core_web_sm
	poetry install


## Initialize directories
init:
	mkdir -p data/bibcodes data/text data/latex

build:
	poetry export -f requirements.txt --output requirements.txt & \
	docker compose build

## Collect SOHO paper metadata.
collect:
	$(py) paper_data_linking/data/get_paper_metadata.py


py=python

# External Data
soho_instruments=data/external/soho_multilabel.csv

# Services
bibcode_service=paper_data_linking/data/bibcode_service.py
metadata_script=paper_data_linking/data/metadata_service.py
url_service=paper_data_linking/data/url_service.py
pdf_download_service=paper_data_linking/data/pdf_download_service.py
processor_service=paper_data_linking/data/processor_service.py
analyzer_script=src/predict.py


# Bibcode output paths
soho_bibcodes=data/raw/bibcodes/soho_bibcodes.txt
cosmo_bibcodes=data/raw/bibcodes/cosmo_bibcodes.txt
solar_bibcodes=data/raw/bibcodes/solar_bibcodes.txt
wind_bibcodes=data/raw/bibcodes/wind_bibcodes.txt


# Metadata output paths
soho_metadata=data/raw/metadata/soho_metadata.jsonl
cosmo_metadata=data/raw/metadata/cosmo_metadata.jsonl
solar_metadata=data/raw/metadata/solar_metadata.jsonl
wind_metadata=data/raw/metadata/wind_metadata.jsonl


# Transformed links output paths
soho_transformed_links=data/raw/links/soho_transformed_links.jsonl
cosmo_transformed_links=data/raw/links/cosmo_transformed_links.jsonl
solar_transformed_links=data/raw/links/solar_transformed_links.jsonl
wind_transformed_links=data/raw/links/wind_transformed_links.jsonl



# pdf output dirs sentinels
pdfs_dir=data/raw/pdfs
pdf_output_dir_soho=$(pdfs_dir)/soho
pdf_output_dir_cosmo=$(pdfs_dir)/cosmo
pdf_output_dir_solar=$(pdfs_dir)/solar
pdf_output_dir_wind=$(pdfs_dir)/wind

soho_pdfs=$(pdf_output_dir_soho)/.done
cosmo_pdfs=$(pdf_output_dir_cosmo)/.done
solar_pdfs=$(pdf_output_dir_solar)/.done
wind_pdfs=$(pdf_output_dir_wind)/.done

# processed data output for evaluation
soho_processed_data_output=data/processed/processed_data.jsonl
wind_processed_data_output=data/processed/wind_processed_data.jsonl


# Bibcode collection targets
$(soho_bibcodes):
	$(py) $(bibcode_service) --library_id HLx1YisxRhyufHOCBhs_Gg --output $@

$(cosmo_bibcodes):
	$(py) $(bibcode_service) --query cosmology --output $@

$(solar_bibcodes):
	$(py) $(bibcode_service) --query solar --output $@

$(wind_bibcodes):
	$(py) $(bibcode_service) --library_id $(WIND_LIB_ID) --output $@


# Metadata collection targets with dependencies
$(soho_metadata): $(soho_bibcodes)
	$(py) $(metadata_script) $< $@

$(cosmo_metadata): $(cosmo_bibcodes)
	$(py) $(metadata_script) $< $@

$(solar_metadata): $(solar_bibcodes)
	$(py) $(metadata_script) $< $@

$(wind_metadata): $(wind_bibcodes)
	$(py) $(metadata_script) $< $@


# URL Transformation targets with dependencies
$(soho_transformed_links): $(soho_metadata)
	$(py) $(url_service) $< $@

$(cosmo_transformed_links): $(cosmo_metadata)
	$(py) $(url_service) $< $@

$(solar_transformed_links): $(solar_metadata)
	$(py) $(url_service) $< $@

$(wind_transformed_links): $(wind_metadata)
	$(py) $(url_service) $< $@


# PDF download targets with dependencies
$(soho_pdfs): $(soho_transformed_links)
	$(py) $(pdf_download_service) --input $< --output-dir $(pdf_output_dir_soho)
	touch $@

$(cosmo_pdfs): $(cosmo_transformed_links)
	$(py) $(pdf_download_service) --input $< --output-dir $(pdf_output_dir_cosmo)
	touch $@

$(solar_pdfs): $(solar_transformed_links)
	$(py) $(pdf_download_service) --input $< --output-dir $(pdf_output_dir_solar)
	touch $@

$(wind_pdfs): $(wind_transformed_links)
	$(py) $(pdf_download_service) --input $< --output-dir $(pdf_output_dir_wind)
	touch $@


# ProcessorService target
topics='{"soho": {"path": "$(soho_transformed_links)", "label": "positive"}, "solar": {"path": "$(solar_transformed_links)", "label": "negative"}, "cosmo": {"path": "$(cosmo_transformed_links)", "label": "negative"}}'
$(soho_processed_data_output): $(soho_transformed_links) $(cosmo_transformed_links) $(solar_transformed_links) $(processor_service)
	$(py) $(processor_service) \
		--csv-path $(soho_instruments) \
		--topics $(topics) \
		--pdfs-base-path $(pdfs_dir) \
		--pubdate-cutoff $(PUBDATE_CUTOFF) \
	    --output-path $(soho_processed_data_output) \
	    --sample-size 100


wind_topics='{"wind": {"path": "$(wind_transformed_links)", "label": "positive"}, "solar": {"path": "$(solar_transformed_links)", "label": "negative"}, "cosmo": {"path": "$(cosmo_transformed_links)", "label": "negative"}}'
$(wind_processed_data_output): $(wind_transformed_links) $(cosmo_transformed_links) $(solar_transformed_links) $(processor_service)
	$(py) $(processor_service) \
		--topics $(wind_topics) \
		--pdfs-base-path $(pdfs_dir) \
		--pubdate-cutoff $(PUBDATE_CUTOFF) \
		--output-path $(wind_processed_data_output) \
		--sample-size 1000 \
		--no-enriched-only

analysis_output=data/analysis/analysis_results.jsonl
analysis_sentinel=data/analysis/.done
analyzer_config=default_config/soho_config_alter.yaml
# Analyzer target
$(analysis_sentinel): $(soho_processed_data_output)
	$(py) $(analyzer_script) \
		$(soho_processed_data_output) \
		$(analysis_output) \
		$(analyzer_config) \
		--num-workers 4


wind_analysis_output=data/analysis/wind_analysis_results.jsonl
wind_analysis_sentinel=data/analysis/wind.done
wind_analyzer_config=default_config/wind.yaml  # Assuming a separate config for WIND

$(wind_analysis_sentinel): $(wind_processed_data_output)
	$(py) $(analyzer_script) \
        $(wind_processed_data_output) \
        $(wind_analysis_output) \
        $(wind_analyzer_config) \
        --num-workers 1

score_output = data/score/.done
# Scoring target
$(score_output): $(analysis_output) $(soho_processed_data_output)
	python src/merge_and_score.py $(analysis_output) $(soho_processed_data_output)
	touch $(score_output)

wind_score_output = data/score/wind.done
$(wind_score_output): $(wind_analysis_output) $(wind_processed_data_output)
	python src/merge_and_score.py $(wind_analysis_output) $(wind_processed_data_output)
	touch $(wind_score_output)


# Phony targets for collecting all
.PHONY: bibcodes metadata, transform_links pdfs, process, analyze score

bibcodes: $(soho_bibcodes) $(cosmo_bibcodes) $(solar_bibcodes) $(wind_bibcodes)
metadata: $(soho_metadata) $(cosmo_metadata) $(solar_metadata) $(wind_metadata)
transform_links: $(soho_transformed_links) $(cosmo_transformed_links) $(solar_transformed_links) $(wind_transformed_links)
pdfs: $(soho_pdfs) $(cosmo_pdfs) $(solar_pdfs) $(wind_pdfs)
process: $(soho_processed_data_output) $(wind_processed_data_output)
analyze: $(analysis_sentinel) $(wind_analysis_sentinel)
score: $(score_output) $(wind_score_output)



## Transform URLs
transform:
	$(py) paper_data_linking/data/transform_urls.py


## Download paper pdf text
download:
	$(py) paper_data_linking/data/download_text.py


## Segment the papers into chunks for LLM processing.
segment:
	$(py) paper_data_linking/analysis/segment.py


## Make publisher bar plot to see how many are open.
viz-pub:
	$(py) paper_data_linking/analysis/viz.py


## prepare test data
prepare:
	python src/prepare.py --in_multilabel data/external/soho_multilabel.csv --data_dir data/raw/ --out_jsonl data/processed/papers_labeled.jsonl

## predict using papers from test data, reads from files on comp
predict:
	python src/predict.py data/processed/papers_labeled.jsonl data/processed/new_results.jsonl default_config/soho_config_alter.yaml

## Evaluate the test data
eval:
	python src/eval.py --processed_data_dir data/processed/ --output_dir reports/

## Download and unzip the chromedriver for your version
update-chromedriver:
	export version_str=$$($(CHROME_PATH) --version); \
	export version_array=($${version_str}); \
	export version=$${version_array[2]}; \
	echo $${version}; \
	export major_version="$$(cut -d'.' -f1 <<<"$${version}")"; \
	echo $${major_version}; \
	export version_url="$$(curl $(CHROMEDRIVER_URL)/LATEST_RELEASE_$${major_version})"; \
	export driver_url=$(CHROMEDRIVER_URL)/$${version_url}/$(OS_PATH); \
	echo $${driver_url}; \
	wget $${driver_url}; \
	unzip chromedriver_mac64.zip; \
	rm chromedriver_mac64.zip; \


ONNX_URL=https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz
## get onnx models
onnx:
	cd models/ && \
	wget $(ONNX_URL) && \
	tar -xzvf onnx.tar.gz


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
