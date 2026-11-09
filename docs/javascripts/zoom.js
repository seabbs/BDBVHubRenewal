// Drag-to-zoom shared by every interactive chart. Drag across the plot to
// zoom the x axis (y rescales to what is visible), drag on the y axis to
// zoom y alone, and double-click or press Reset to undo. A chart keeps one
// zoom object, so zoom survives metric tabs and model ticks: x is kept
// for the chart and y for each metric (the `ykey` a chart passes).
(function () {
  "use strict";
  var count = 0;
  var el = window.bdbvToggles.el;


  // controls: where the Reset button goes. onChange redraws the chart.
  function create(controls, onChange) {
    var z = { x: null, y: {}, onChange: onChange };
    z.button = window.bdbvToggles.h("button",
      { type: "button", class: "zoom-reset", hidden: "" }, controls, "Reset zoom");
    z.button.addEventListener("click", function () { reset(z); });
    return z;
  }

  function reset(z) {
    z.x = null;
    z.y = {};
    z.onChange();
  }

  // The x range to draw: the zoomed one or the full [x0, x1].
  function xRange(z, x0, x1) { return z.x ? z.x.slice() : [x0, x1]; }

  // Whether x is in the range drawn, to rescale y to the visible data.
  function visible(z, x) { return !z.x || (x >= z.x[0] && x <= z.x[1]); }

  // The y range to draw for ykey: the zoomed one or [lo, hi].
  function yRange(z, lo, hi, ykey) {
    var y = z.y[ykey || ""];
    return y ? y.slice() : [lo, hi];
  }

  // Clip the data marks to the plot area g: {L, R, T, B, W, H}.
  function clip(svg, g) {
    count += 1;
    var id = "zoomclip" + count;
    var cp = el("clipPath", { id: id }, el("defs", {}, svg));
    el("rect", { x: g.L, y: g.T, width: g.W - g.L - g.R, height: g.H - g.T - g.B }, cp);
    return el("g", { "clip-path": "url(#" + id + ")" }, svg);
  }

  // Let the reader brush the chart just drawn. g has the plot area and the
  // ranges drawn, x: [x0, x1] and y: [y0, y1], linear in pixels.
  function attach(z, svg, g, ykey) {
    z.button.hidden = !(z.x || z.y[ykey || ""]);
    svg.classList.add("zoomable");
    function toSvg(e) {
      var p = svg.createSVGPoint();
      p.x = e.clientX;
      p.y = e.clientY;
      return p.matrixTransform(svg.getScreenCTM().inverse());
    }
    function xData(px) { return g.x[0] + (px - g.L) / (g.W - g.L - g.R) * (g.x[1] - g.x[0]); }
    function yData(py) { return g.y[0] + (g.H - g.B - py) / (g.H - g.T - g.B) * (g.y[1] - g.y[0]); }
    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    var start = null, box = null, axis = null;
    svg.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      var p = toSvg(e);
      if (p.y > g.H - g.B + 2 || p.x > g.W - g.R) return;
      axis = p.x < g.L ? "y" : "x";
      start = p;
      box = el("rect", { class: "zoom-brush" }, svg);
      svg.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    svg.addEventListener("pointermove", function (e) {
      if (!start) return;
      var p = toSvg(e);
      if (axis === "x") {
        var a = clamp(Math.min(start.x, p.x), g.L, g.W - g.R);
        var b = clamp(Math.max(start.x, p.x), g.L, g.W - g.R);
        box.setAttribute("x", a); box.setAttribute("width", b - a);
        box.setAttribute("y", g.T); box.setAttribute("height", g.H - g.T - g.B);
      } else {
        var c = clamp(Math.min(start.y, p.y), g.T, g.H - g.B);
        var d = clamp(Math.max(start.y, p.y), g.T, g.H - g.B);
        box.setAttribute("y", c); box.setAttribute("height", d - c);
        box.setAttribute("x", g.L); box.setAttribute("width", g.W - g.L - g.R);
      }
    });
    svg.addEventListener("pointerup", function (e) {
      if (!start) return;
      var p = toSvg(e);
      box.remove();
      var s = start;
      start = null;
      if (axis === "x" && Math.abs(p.x - s.x) > 6) {
        var lo = xData(clamp(Math.min(s.x, p.x), g.L, g.W - g.R));
        var hi = xData(clamp(Math.max(s.x, p.x), g.L, g.W - g.R));
        z.x = [lo, hi];
        z.y[ykey || ""] = null;
        z.onChange();
      } else if (axis === "y" && Math.abs(p.y - s.y) > 6) {
        var top = yData(clamp(Math.min(s.y, p.y), g.T, g.H - g.B));
        var bot = yData(clamp(Math.max(s.y, p.y), g.T, g.H - g.B));
        z.y[ykey || ""] = [bot, top];
        z.onChange();
      }
    });
    svg.addEventListener("dblclick", function () { reset(z); });
  }

  // A round tick step giving about five ticks over span.
  function nice(span) {
    var p = Math.pow(10, Math.floor(Math.log10(span / 5)));
    var f = span / 5 / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }

  window.bdbvZoom = {
    create: create, xRange: xRange, visible: visible, yRange: yRange,
    clip: clip, attach: attach, nice: nice,
  };
})();
