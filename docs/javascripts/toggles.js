// Model tick boxes shared by the score and zone charts: a checkbox and a
// line swatch per model, in each model's light or dark colour and dash, and
// a redraw when the site theme changes.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function h(tag, attrs, parent, text) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function mode() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "light";
  }

  // A short line in a model's dash, for the legend. Returns the line.
  function swatch(parent, dash) {
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "scorechart-swatch");
    svg.setAttribute("viewBox", "0 0 24 8");
    var line = document.createElementNS(NS, "line");
    [["x1", 0], ["y1", 4], ["x2", 24], ["y2", 4], ["stroke-width", 3],
      ["stroke-dasharray", dash || ""]].forEach(function (a) {
      line.setAttribute(a[0], a[1]);
    });
    svg.appendChild(line);
    parent.appendChild(svg);
    return line;
  }

  // models: [{name, light, dark, dash, on}]. Returns {on, colour(i),
  // dash(i), paint()}:
  // `on` holds each model's state and onChange runs after a tick.
  function legend(parent, models, onChange) {
    var root = h("div", { class: "scorechart-legend" }, parent);
    var on = models.map(function (m) { return m.on; });
    var swatches = models.map(function (m, i) {
      var l = h("label", { class: "scorechart-key" }, root);
      var cb = h("input", { type: "checkbox" }, l);
      cb.checked = m.on;
      var sw = swatch(l, m.dash);
      h("span", {}, l, m.name);
      cb.addEventListener("change", function () { on[i] = cb.checked; onChange(); });
      return sw;
    });
    return {
      on: on,
      colour: function (i) { return models[i][mode()]; },
      dash: function (i) { return models[i].dash || ""; },
      paint: function () {
        swatches.forEach(function (sw, i) { sw.setAttribute("stroke", models[i][mode()]); });
      },
    };
  }

  function onTheme(draw) {
    new MutationObserver(draw).observe(document.body,
      { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }

  // A Bands / Trajectories select, Bands first. Its value is "bands" or
  // "traj". Without trajectories it is not shown and stays on bands.
  function viewSelect(controls, has, onChange) {
    if (!has) return { value: "bands" };
    var sel = h("select", {}, h("label", {}, controls, "Show "));
    h("option", { value: "bands" }, sel, "Bands");
    h("option", { value: "traj" }, sel, "Trajectories");
    sel.addEventListener("change", onChange);
    return sel;
  }

  window.bdbvToggles = { h: h, legend: legend, onTheme: onTheme, viewSelect: viewSelect };
})();
