function displayJson(jsonData) {
  // Get the JSON container
  const jsonContainer = $("#json-container");

  // Display the JSON data
  jsonContainer.jsonViewer(jsonData);
}

class PDFHighlighter {
  constructor() {
    this.pdfFileInput = document.getElementById("pdf-file-input");
    this.highlightButton = document.getElementById("highlight-button");
    this.pdfContainer = document.getElementById("pdf-container");
    this.textContainer = document.getElementById("text-container");
    this.analysisContainer = document.getElementById("analysis-container");
    this.loadingIndicator = document.getElementById("loading-indicator");
    this.progressMessageElement = document.getElementById("progress-message");
    this.configSelector = document.getElementById("config-selector");
    this.uploadedFile = null;
    this.scale = null;

    this.configSelector.addEventListener("change", (e) => {
      // Re-enable the "Analyze" button whenever the classifier is changed
      if (e.target.value && this.uploadedFile) {
        this.highlightButton.disabled = false;
      } else {
        // Disable the "Analyze" button if no classifier is selected (optional)
        this.highlightButton.disabled = true;
      }
    });

    this.pdfFileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (file) {
        try {
          this.uploadedFile = file;
          this.loadingIndicator.style.visibility = "visible";
          this.highlightButton.disabled = true;
          this.pdfContainer.innerHTML = "";
          this.analysisContainer.textContent = "";
          // Initialize without highlights, hence []
          await this.renderPDFWithHighlights([]);
          this.highlightButton.disabled = false;
        } catch (error) {
          console.error("Error during processing:", error);
          alert("An error occurred during processing. Please try again.");
        } finally {
          this.loadingIndicator.style.visibility = "hidden";
          this.highlightButton.disabled = false;
        }
      }
    });

    this.highlightButton.addEventListener("click", async () => {
      if (this.uploadedFile) {
        try {
          this.loadingIndicator.style.visibility = "visible";
          this.highlightButton.disabled = true;
          this.progressMessageElement.innerText = `Processing...`;
          this.progressMessageElement.innerText = `parsing⏳ › embedding⏳ › analyzing⏳\nStarted...`;

          // Initialize loading messages
          this.textContainer.innerHTML = "Loading...";
          this.analysisContainer.innerHTML = "Loading...";
          displayJson({});

          // Get the selected config
          const selectedConfig = this.configSelector.value;
          // Create and send the form data
          const formData = new FormData();
          formData.append("file", this.uploadedFile);
          formData.append("config_file", selectedConfig);

          // Make the POST request to start the Celery task
          const taskResponse = await axios.post(
            "/api/highlight_pdf",
            formData,
            {
              headers: {
                "Content-Type": "multipart/form-data",
              },
            },
          );

          // Extract the task ID from the response
          const taskId = taskResponse.data.task_id;

          // Poll the task status every 5 seconds
          const intervalId = setInterval(async () => {
            const statusResponse = await axios.get(`/api/task/${taskId}`);

            if (statusResponse.data.task_status === "SUCCESS") {
              // If the task is done, clear the interval and process the results
              clearInterval(intervalId);

              // Extract the task result
              const result = statusResponse.data.task_result;
              const text = result.text;
              const highlights = result.highlights;
              const analysis = result.analysis;
              const jsonData = result.data;

              this.pdfContainer.innerHTML = "";
              await this.renderPDFWithHighlights(highlights);

              this.textContainer.innerHTML = text;
              this.analysisContainer.innerHTML = marked.parse(analysis);
              this.addJumpButtons(highlights);

              displayJson(jsonData);

              // Render viz (right now assuming using only one plugin, hence [0])
              const analyzer = jsonData.results[0].analyzer;
              const analyzerData = jsonData.results[0];
              renderD3Visualization(analyzerData, analyzer);

              // this.uploadedFile = null;

              this.loadingIndicator.style.visibility = "hidden";
              this.highlightButton.disabled = false;
              this.progressMessageElement.innerText = `parsing✅ › embedding✅ › analyzing✅\nComplete🎉`;
            } else if (statusResponse.data.task_status === "PROGRESS") {
              // Update the progress message with the current status
              const progress = statusResponse.data.task_result.current;
              const status = statusResponse.data.task_result.status;
              this.progressMessageElement.innerText = `${progress}\n${status}`;
            } else if (statusResponse.data.task_status === "FAILURE") {
              // If the task failed, clear the interval and alert the user
              clearInterval(intervalId);

              console.error(
                "Error during processing:",
                statusResponse.data.task_result,
              );
              alert("An error occurred during processing. Please try again.");

              this.loadingIndicator.style.visibility = "hidden";
              this.highlightButton.disabled = false;
              this.progressMessageElement.innerText = `parsing❓ › embedding❓ › analyzing\nFailed❌`;
            }
          }, 1000);
        } catch (error) {
          console.error("Error during processing:", error);
          alert("An error occurred during processing. Please try again.");

          this.loadingIndicator.style.visibility = "hidden";
          this.highlightButton.disabled = false;
        }
      }
    });
  }

  async renderPDFWithHighlights(highlights) {
    const file = this.uploadedFile;
    const pdfData = new Uint8Array(await file.arrayBuffer());
    const pdfDoc = await pdfjsLib.getDocument({ data: pdfData }).promise;

    for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
      const page = await pdfDoc.getPage(pageNum);
      const canvas = document.createElement("canvas");
      canvas.className = "canvas-element";
      const context = canvas.getContext("2d");

      const containerWidth = this.pdfContainer.clientWidth;
      const unscaledViewport = page.getViewport({ scale: 1 });
      const scale = containerWidth / unscaledViewport.width;
      const viewport = page.getViewport({ scale });
      this.scale = scale;

      canvas.width = viewport.width;
      canvas.height = viewport.height;
      this.pdfContainer.appendChild(canvas);

      await page.render({ canvasContext: context, viewport }).promise;

      const pageHighlights = highlights.filter(
        (highlight) => highlight.page === pageNum,
      );

      context.globalCompositeOperation = "multiply";

      const intensityValues = highlights.map(
        (highlight) => highlight.intensity,
      );
      const minCosineSimilarity = Math.min(...intensityValues);
      const maxCosineSimilarity = Math.max(...intensityValues);

      for (const highlight of pageHighlights) {
        const intensity = this.transformIntensity(
          highlight.intensity,
          minCosineSimilarity,
          maxCosineSimilarity,
        );
        // Check if highlight has a color property, default to yellow if not
        const color = highlight.color
          ? `rgba(${highlight.color[0]}, ${highlight.color[1]}, ${highlight.color[2]}, 0.5)`
          : `rgba(255, 255, 0, 0.5)`;
        context.fillStyle = color;

        const rect = highlight.rect;
        context.fillRect(
          rect.x * scale,
          rect.y * scale,
          rect.width * scale,
          rect.height * scale,
        );
      }
    }
  }

  transformIntensity(
    cosineSimilarity,
    minCosineSimilarity,
    maxCosineSimilarity,
  ) {
    const exponent = 3;
    const normalizedValue =
      (cosineSimilarity - minCosineSimilarity) /
      (maxCosineSimilarity - minCosineSimilarity);
    const transformedValue = Math.pow(normalizedValue, exponent);
    return transformedValue;
  }

  jumpToHighlight(highlight) {
    const pageNum = highlight.page;
    const rect = highlight.rect;

    const canvasElements = document.querySelectorAll(".canvas-element");
    const canvasElement = canvasElements[pageNum - 1];

    if (canvasElement) {
      const pdfContainer = document.getElementById("pdf-container");
      const scrollOffset = canvasElement.offsetTop - pdfContainer.offsetTop;

      pdfContainer.scrollTo({
        top: scrollOffset,
        behavior: "smooth",
      });

      const scale = this.scale;
      const jumpToRect = {
        x: rect.x * scale,
        y: rect.y * scale,
        width: rect.width * scale,
        height: rect.height * scale,
      };

      this.highlightRectangle(canvasElement, jumpToRect);
    }
  }

  highlightRectangle(canvasElement, rect) {
    const context = canvasElement.getContext("2d");
    const previousState = context.getImageData(
      0,
      0,
      canvasElement.width,
      canvasElement.height,
    );

    context.strokeStyle = "red";
    context.lineWidth = 2;
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);

    // Disable the buttons
    const buttons = document.getElementsByClassName("jump-button");
    for (let i = 0; i < buttons.length; i++) {
      buttons[i].disabled = true;
    }

    setTimeout(() => {
      context.putImageData(previousState, 0, 0);

      // Enable the buttons
      for (let i = 0; i < buttons.length; i++) {
        buttons[i].disabled = false;
      }
    }, 1500);
  }

  addJumpButtons(highlights) {
    highlights.forEach((highlight, index) => {
      const jumpButton = document.createElement("button");
      jumpButton.className = "jump-button";
      jumpButton.textContent = `Jump to Excerpt ${index + 1}`;
      jumpButton.addEventListener("click", () =>
        this.jumpToHighlight(highlight),
      );
      this.analysisContainer.appendChild(jumpButton);
    });
  }

  async handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) {
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post("/api/upload_pdf", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    this.pdfContainer.innerHTML = "";
    this.uploadedFile = file;
  }
}

const pdfHighlighter = new PDFHighlighter();
document
  .getElementById("pdf-file-input")
  .addEventListener(
    "change",
    pdfHighlighter.handleFileSelect.bind(pdfHighlighter),
  );
