// Model tick boxes shared by the score and zone charts: a checkbox and a
// swatch per model, in each model's light or dark colour, and a redraw
// when the site theme changes.
(function () {
  "use strict";

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

  // models: [{name, light, dark, on}]. Returns {on, colour(i), paint()}:
  // `on` holds each model's state and onChange runs after a tick.
  function legend(parent, models, onChange) {
    var root = h("div", { class: "scorechart-legend" }, parent);
    var on = models.map(function (m) { return m.on; });
    var swatches = models.map(function (m, i) {
      var l = h("label", { class: "scorechart-key" }, root);
      var cb = h("input", { type: "checkbox" }, l);
      cb.checked = m.on;
      var sw = h("span", { class: "scorechart-swatch" }, l);
      h("span", {}, l, m.name);
      cb.addEventListener("change", function () { on[i] = cb.checked; onChange(); });
      return sw;
    });
    return {
      on: on,
      colour: function (i) { return models[i][mode()]; },
      paint: function () {
        swatches.forEach(function (sw, i) { sw.style.background = models[i][mode()]; });
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
