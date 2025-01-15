// static/js/vizRenderer.js

function renderD3Visualization(data, plugin) {
    const container = d3.select("#d3-container");
    container.selectAll("*").remove();  // Clear any existing visualizations

    switch(plugin) {
        case "TimeRange":
            renderTimeExtractionViz(container, data.data["Identified Time Ranges"]); // Pass only the relevant data for this visualization
            break;
        // Add other cases as you create more visualizations
    }
}

