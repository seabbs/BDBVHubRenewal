// Model tick boxes shared by the score and zone charts: a checkbox and a
// line swatch per model, in each model's light or dark colour and dash, and
// a redraw when the site theme changes.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  // An SVG element.
  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n);
    return n;
  }

  // The JSON at src, failing on an HTTP error.
  function getJSON(src) {
    return fetch(src).then(function (r) {
      if (!r.ok) throw new Error(src + ": " + r.status);
      return r.json();
    });
  }

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

  // Bars cannot show a dash, so a dashed model's bars are hatched in its
  // colour. Adds the pattern to the chart's <defs> and returns the fill.
  var hatches = 0;
  function hatch(svg, colour) {
    var defs = svg.querySelector("defs");
    if (!defs) defs = svg.insertBefore(document.createElementNS(NS, "defs"), svg.firstChild);
    var id = "bdbv-hatch-" + (hatches += 1);
    var p = document.createElementNS(NS, "pattern");
    [["id", id], ["patternUnits", "userSpaceOnUse"], ["width", 6], ["height", 6],
      ["patternTransform", "rotate(45)"]].forEach(function (a) { p.setAttribute(a[0], a[1]); });
    var line = document.createElementNS(NS, "line");
    [["x1", 0], ["y1", 0], ["x2", 0], ["y2", 6], ["stroke", colour],
      ["stroke-width", 3]].forEach(function (a) { line.setAttribute(a[0], a[1]); });
    p.appendChild(line);
    defs.appendChild(p);
    return "url(#" + id + ")";
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

  // A Full screen button that shows target alone, and back, where the
  // browser can show an element full screen.
  function fullscreen(controls, target) {
    if (!document.fullscreenEnabled) return;
    var b = h("button", { type: "button", class: "zoom-reset fullscreen" }, controls, "Full screen");
    b.addEventListener("click", function () {
      if (document.fullscreenElement) document.exitFullscreen();
      else target.requestFullscreen();
    });
    document.addEventListener("fullscreenchange", function () {
      b.textContent = document.fullscreenElement === target ? "Exit full screen" : "Full screen";
    });
  }

  // A "Scored on" select of a widget's targets (data-targets: target to
  // summary URL), showing value. onChange(target) runs on a change. With
  // one target there is nothing to pick and no select.
  var TARGETS = { latest: "Latest release", complete: "Complete weeks" };
  function targetSelect(controls, targets, value, onChange) {
    var keys = Object.keys(targets);
    if (keys.length < 2) return;
    var sel = h("select", {}, h("label", {}, controls, "Scored on "));
    keys.forEach(function (k) { h("option", { value: k }, sel, TARGETS[k]); });
    sel.value = value;
    sel.addEventListener("change", function () { onChange(sel.value); });
  }

  // A Map tick box, ticked when on. onChange(ticked) runs on a change.
  function mapToggle(controls, on, onChange) {
    var l = h("label", { class: "map-toggle" }, controls);
    var cb = h("input", { type: "checkbox" }, l);
    cb.checked = on;
    h("span", {}, l, " Map");
    cb.addEventListener("change", function () { onChange(cb.checked); });
    return cb;
  }

  window.bdbvToggles = { h: h, el: el, getJSON: getJSON, legend: legend, hatch: hatch, onTheme: onTheme,
    viewSelect: viewSelect, fullscreen: fullscreen, mapToggle: mapToggle,
    targetSelect: targetSelect };
})();
