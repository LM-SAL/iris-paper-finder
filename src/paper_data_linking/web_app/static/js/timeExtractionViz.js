function parseTime(timeString) {
  if (timeString.length === 4) {
    return new Date(timeString, 0, 1);
  } else if (timeString.length === 7) {
    return new Date(timeString + "-01");
  } else {
    return new Date(timeString);
  }
}

function renderTimeExtractionViz(container, data) {
  // Data
  var timeRanges = data
    .map((entry) => {
      return {
        dataSource: entry.label.data_source,
        timeRange: entry.label.time_range,
        support: [],
      };
    })
    .filter((entry) => !entry.timeRange.includes("UNKNOWN"));

  // Dynamic SVG setup based on container width
  var containerRect = container.node().getBoundingClientRect();
  var containerWidth = containerRect.width;

  var margin = { top: 70, right: 50, bottom: 50, left: 50 },
    width = containerWidth - margin.left - margin.right, // Use container width
    height = 350 - margin.top - margin.bottom; // Increase height to 400 from 300

  var svg = container
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  svg
    .append("defs")
    .append("clipPath")
    .attr("id", "clip")
    .append("rect")
    .attr("width", width)
    .attr("height", height);

  var timeContextLabel = svg
    .append("text")
    .attr("class", "time-context-label")
    .attr("x", width / 2)
    .attr("y", height + 40) // 40 pixels below the x-axis
    .attr("text-anchor", "middle");

  // Calculate Min and Max dates from the data
  var allDates = timeRanges.flatMap((d) => d.timeRange.map(parseTime));
  var minDate = new Date(Math.min.apply(null, allDates));
  var maxDate = new Date(Math.max.apply(null, allDates));

  // Time scale
  var x = d3.scaleTime().domain([minDate, maxDate]).range([0, width]);

  // Add a border around the graph area
  svg
    .append("rect")
    .attr("x", 0) // x-position is 0
    .attr("y", 0) // y-position is 0
    .attr("width", width) // set to the width of the graph
    .attr("height", height) // set to the height of the graph
    .style("stroke", "black") // border color
    .style("fill", "none") // no fill
    .style("stroke-width", 1); // border width

  var xAxis = d3.axisBottom(x);

  // Add vertical grid lines
  svg
    .append("g")
    .attr("class", "grid")
    .attr("transform", "translate(0," + height + ")")
    .call(
      d3
        .axisBottom(x)
        .ticks(20) // Adjust the number of ticks for grid lines
        .tickSize(-height)
        .tickFormat(""),
    )
    .attr("stroke-opacity", "0.2");

  var gX = svg
    .append("g")
    .attr("transform", "translate(0," + height + ")")
    .attr("class", "axis")
    .call(xAxis);

  // Tooltip
  var tooltip = d3.select("body").append("div").attr("class", "tooltip");

  // Create a brush
  var brush = d3
    .brushX()
    .extent([
      [0, 0],
      [width, height],
    ])
    .on("end", brushed);

  // Append brush to SVG
  var gBrush = svg.append("g").attr("class", "brush").call(brush);

  // Add double-click event to reset brush
  gBrush.on("dblclick", function () {
    x.domain([minDate, maxDate]);
    svg.select(".axis").call(xAxis);
    renderBarsAndText();
  });

  // Brush event handler
  function brushed() {
    var selection = d3.event.selection;
    if (selection) {
      var [x0, x1] = selection;
      var newXDomain = [x.invert(x0), x.invert(x1)];

      // Update your x scale
      x.domain(newXDomain);

      // Update your axis and re-render your data
      svg.select(".axis").call(xAxis);

      // ... re-render your bars and text
      timeContextLabel.text(
        "Date: " +
          x.domain()[0].toDateString() +
          " - " +
          x.domain()[1].toDateString(),
      );
      renderBarsAndText();
    }
  }

  // Plot data
  var color = d3.scaleOrdinal(d3.schemeCategory10);

  function renderBarsAndText() {
    svg.selectAll(".timeline-bar, .timeline-circle, .text-label").remove(); // Clear previous bars, circles, and labels

    timeRanges.forEach(function (d, i) {
      var y = (i + 1) * 40;
      y = y - 2; // shift up a little bit
      var startDate = parseTime(d.timeRange[0]);
      var endDate = parseTime(d.timeRange[1]);

      // Check for zero-duration events
      if (startDate.getTime() === endDate.getTime()) {
        svg
          .append("circle")
          .attr("class", "timeline-circle") // Added class for later selection
          .attr("cx", x(startDate))
          .attr("cy", y + 5) // center the circle in the middle of where the bar would be
          .attr("r", 5) // radius
          .style("fill", color(i));
      } else {
        svg
          .append("rect")
          .attr("class", "timeline-bar")
          .attr("clip-path", "url(#clip)")
          .style("fill", color(i))
          .attr("x", x(startDate))
          .attr("y", y)
          .attr("width", x(endDate) - x(startDate))
          .attr("height", 10);
      }

      svg
        .append("text")
        .attr("class", "text-label")
        .attr("x", x(startDate))
        .attr("y", y - 10)
        .text(d.dataSource.join(", "));
    });
  }

  // Add this code near the end of your renderTimeExtractionViz function

  var legend = svg
    .append("g")
    .attr("class", "legend")
    .attr("transform", "translate(" + 2 + "," + -50 + ")"); // Shifted closer to the corner

  // Add background box for legend
  legend
    .append("rect")
    .attr("x", -5) // Slightly to the left to encompass items
    .attr("y", -5) // Slightly to the top to encompass items
    .attr("width", 160) // Sufficient width to contain all items and text
    .attr("height", 50) // Sufficient height to contain both items
    .style(
      "fill",
      getComputedStyle(document.documentElement).getPropertyValue(
        "--brand-rose",
      ),
    )
    .style("stroke", "black");

  // Add rectangles for legend
  legend
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", 20)
    .attr("height", 10)
    .style("fill", "grey");

  // Add circles for legend
  legend
    .append("circle")
    .attr("cx", 10)
    .attr("cy", 25)
    .attr("r", 5)
    .style("fill", "grey");

  // Add text for rectangles
  legend
    .append("text")
    .attr("x", 30)
    .attr("y", 10)
    .attr("dy", "0.1em")
    .style("text-anchor", "start")
    .text("Time Range");

  // Add text for circles
  legend
    .append("text")
    .attr("x", 30)
    .attr("y", 30)
    .attr("dy", "0.1em")
    .style("text-anchor", "start")
    .text("Discrete Time");

  // Add title to the top
  svg
    .append("text")
    .attr("x", width / 2)
    .attr("y", -40)
    .attr("text-anchor", "middle")
    .style("font-size", "20px")
    .style("font-weight", "bold")
    .text("Data Sources and Time Ranges");

  timeContextLabel.text(
    "Date: " + minDate.toDateString() + " - " + maxDate.toDateString(),
  );

  renderBarsAndText(); // Initial rendering
}
