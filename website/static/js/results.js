// Results grid: failure rate (collision + off-road per vehicle) of the three
// students for two policy sizes on three test sets, teacher as a dashed line.
// Numbers are Table 1 of the paper (9 seeds, mean +- SE). Bars rise when the
// grid scrolls into view; hovering a bar shows its value.

const POLICIES = [
  { label: "MLP", color: "#2a78d6" },
  { label: "Unconstrained DiffMPC", color: "#eb6834" },
  { label: "Constrained DiffMPC", color: "#eda100" },
];
const TEST_SETS = ["CARLA", "nuPlan", "Waymo"];
// size -> test set -> { teacher, students: [[mean, se] per policy] }
const FAILURE_RATES = {
  "4x512": {
    CARLA: { teacher: 0.245, students: [[0.226, 0.004], [0.224, 0.003], [0.180, 0.003]] },
    nuPlan: { teacher: 0.181, students: [[0.205, 0.006], [0.220, 0.005], [0.180, 0.006]] },
    Waymo: { teacher: 0.298, students: [[0.299, 0.007], [0.280, 0.008], [0.261, 0.008]] },
  },
  "4x1024": {
    CARLA: { teacher: 0.052, students: [[0.050, 0.001], [0.052, 0.002], [0.053, 0.001]] },
    nuPlan: { teacher: 0.039, students: [[0.040, 0.001], [0.044, 0.003], [0.035, 0.001]] },
    Waymo: { teacher: 0.059, students: [[0.059, 0.002], [0.059, 0.002], [0.046, 0.002]] },
  },
};
const ROW_AXIS = {
  "4x512": { max: 0.35, step: 0.05, digits: 2 },
  "4x1024": { max: 0.07, step: 0.01, digits: 2 },
};
// Row titles shown on the page; the keys above are the network sizes.
const ROW_TITLES = { "4x512": "Small teacher", "4x1024": "Large teacher" };
const SVG_NS = "http://www.w3.org/2000/svg";
const PANEL_WIDTH = 300;
const PANEL_HEIGHT = 210;
const PLOT_LEFT = 46;
const PLOT_RIGHT = 290;
const PLOT_TOP = 12;
const PLOT_BOTTOM = 196;
const BAR_SLOT_FRACTION = 0.72;
const BAR_DELAY_MS = 110;

function svgElement(tag, attributes) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, value);
  }
  return element;
}

function drawPanel(size, testSet, panelIdx, tooltip) {
  const { teacher, students } = FAILURE_RATES[size][testSet];
  const axis = ROW_AXIS[size];
  const yOf = (rate) => PLOT_BOTTOM - (rate / axis.max) * (PLOT_BOTTOM - PLOT_TOP);
  const svg = svgElement("svg", {
    viewBox: `0 0 ${PANEL_WIDTH} ${PANEL_HEIGHT}`,
    class: "results-panel",
    role: "img",
    "aria-label": `${ROW_TITLES[size]} on ${testSet}: teacher ${teacher}, ` +
      POLICIES.map((policy, idx) => `${policy.label} ${students[idx][0]}`).join(", "),
  });

  const tickCount = Math.round(axis.max / axis.step);
  for (let tickIdx = 0; tickIdx <= tickCount; tickIdx++) {
    const rate = tickIdx * axis.step;
    const y = yOf(rate);
    svg.appendChild(svgElement("line", { x1: PLOT_LEFT, x2: PLOT_RIGHT, y1: y, y2: y, class: "grid-line" }));
    const label = svgElement("text", { x: PLOT_LEFT - 6, y: y + 4, class: "tick-label" });
    label.textContent = rate.toFixed(axis.digits);
    svg.appendChild(label);
  }
  svg.appendChild(svgElement("line", { x1: PLOT_LEFT, x2: PLOT_LEFT, y1: PLOT_TOP, y2: PLOT_BOTTOM, class: "axis-line" }));
  svg.appendChild(svgElement("line", { x1: PLOT_LEFT, x2: PLOT_RIGHT, y1: PLOT_BOTTOM, y2: PLOT_BOTTOM, class: "axis-line" }));

  const slotWidth = (PLOT_RIGHT - PLOT_LEFT) / POLICIES.length;
  const barWidth = slotWidth * BAR_SLOT_FRACTION;
  POLICIES.forEach((policy, policyIdx) => {
    const [mean, se] = students[policyIdx];
    const x = PLOT_LEFT + slotWidth * policyIdx + (slotWidth - barWidth) / 2;
    const delayMs = (panelIdx * POLICIES.length + policyIdx) * BAR_DELAY_MS / 2;
    const bar = svgElement("rect", {
      x, y: yOf(mean), width: barWidth, height: PLOT_BOTTOM - yOf(mean),
      fill: policy.color, class: "bar", style: `transition-delay: ${delayMs}ms`,
    });
    bar.addEventListener("mouseenter", (event) => {
      tooltip.textContent = `${policy.label}: ${mean.toFixed(3)} ± ${se.toFixed(3)}  (teacher ${teacher.toFixed(3)})`;
      tooltip.classList.add("is-visible");
      moveTooltip(tooltip, event);
    });
    bar.addEventListener("mousemove", (event) => moveTooltip(tooltip, event));
    bar.addEventListener("mouseleave", () => tooltip.classList.remove("is-visible"));
    svg.appendChild(bar);

    const centerX = x + barWidth / 2;
    const whisker = svgElement("g", { class: "whisker", style: `transition-delay: ${delayMs + 500}ms` });
    whisker.appendChild(svgElement("line", { x1: centerX, x2: centerX, y1: yOf(mean - se), y2: yOf(mean + se) }));
    whisker.appendChild(svgElement("line", { x1: centerX - 4, x2: centerX + 4, y1: yOf(mean - se), y2: yOf(mean - se) }));
    whisker.appendChild(svgElement("line", { x1: centerX - 4, x2: centerX + 4, y1: yOf(mean + se), y2: yOf(mean + se) }));
    svg.appendChild(whisker);
  });

  const teacherDelayMs = (panelIdx + 1) * POLICIES.length * BAR_DELAY_MS / 2 + 400;
  svg.appendChild(svgElement("line", {
    x1: PLOT_LEFT, x2: PLOT_RIGHT, y1: yOf(teacher), y2: yOf(teacher),
    class: "teacher-line", style: `transition-delay: ${teacherDelayMs}ms`,
  }));
  return svg;
}

function moveTooltip(tooltip, event) {
  tooltip.style.left = `${event.clientX + 12}px`;
  tooltip.style.top = `${event.clientY - 30}px`;
}

function buildResultsGrid(container) {
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  document.body.appendChild(tooltip);

  container.appendChild(document.createElement("div"));
  for (const testSet of TEST_SETS) {
    const heading = document.createElement("div");
    heading.className = "results-column-title";
    heading.textContent = testSet;
    container.appendChild(heading);
  }
  let panelIdx = 0;
  for (const size of Object.keys(FAILURE_RATES)) {
    const rowLabel = document.createElement("div");
    rowLabel.className = "results-row-title";
    rowLabel.innerHTML = `<span>${ROW_TITLES[size]}</span><small>failure rate</small>`;
    container.appendChild(rowLabel);
    for (const testSet of TEST_SETS) {
      container.appendChild(drawPanel(size, testSet, panelIdx, tooltip));
      panelIdx++;
    }
  }

  const legend = document.createElement("div");
  legend.className = "results-legend";
  legend.innerHTML = '<span><i class="legend-dash"></i>Teacher (Self-Play RL)</span>' +
    POLICIES.map((policy) => `<span><i style="background:${policy.color}"></i>${policy.label}</span>`).join("");
  container.after(legend);

  // Rise once, the first time the grid is mostly on screen.
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      container.classList.add("is-revealed");
      observer.disconnect();
    }
  }, { threshold: 0.35 });
  observer.observe(container);
}

document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("results-grid");
  if (container) buildResultsGrid(container);
});
