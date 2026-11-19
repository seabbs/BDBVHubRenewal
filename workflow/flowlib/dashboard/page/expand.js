// Full screen for every card and for the nowcasts view with its controls.
// Each gets a button; it opens the Fullscreen API where the browser has it,
// else a fixed overlay; Esc or the button closes it. Plots drawn by script
// redraw on the resize event sent once the layout has settled after each
// change, and whenever the open element's size changes (ResizeObserver);
// they keep their own state (zoom, selections). `ftFull(el)` is the open
// element holding el, if any, so a chart can fill it.
(function(){
 var open=null,ro=null,t0=null;
 function redraw(){clearTimeout(t0);requestAnimationFrame(function(){requestAnimationFrame(function(){
  window.dispatchEvent(new Event("resize"))})});t0=setTimeout(function(){window.dispatchEvent(new Event("resize"))},250)}
 window.ftFull=function(el){return open&&el&&open.contains(el)?open:null};
 function watch(el){if(ro)ro.disconnect();ro=null;
  if(el&&window.ResizeObserver){var t=null;ro=new ResizeObserver(function(){clearTimeout(t);t=setTimeout(function(){window.dispatchEvent(new Event("resize"))},100)});ro.observe(el)}}
 function close(){if(!open)return;var el=open;open=null;
  el.classList.remove("fs");document.body.classList.remove("fsopen");watch(null);
  var b=el.querySelector(":scope > .expand");if(b){b.textContent="⤢";b.setAttribute("aria-label","Full screen");b.focus()}
  if(document.fullscreenElement&&document.exitFullscreen)document.exitFullscreen().catch(function(){});
  redraw()}
 function show(el){if(open)close();open=el;
  el.classList.add("fs");document.body.classList.add("fsopen");watch(el);
  var b=el.querySelector(":scope > .expand");if(b){b.textContent="✕";b.setAttribute("aria-label","Close full screen");b.focus()}
  if(el.requestFullscreen)el.requestFullscreen().catch(function(){});
  redraw()}
 function add(el){if(el.querySelector(":scope > .expand"))return;
  var b=document.createElement("button");b.type="button";b.className="expand";
  var on=open===el;b.textContent=on?"✕":"⤢";b.setAttribute("aria-label",on?"Close full screen":"Full screen");b.title="Full screen (Esc closes)";
  b.addEventListener("click",function(){open===el?close():show(el)});
  el.insertBefore(b,el.firstChild)}
 // Cards inside the nowcasts view open with the whole view.
 function scan(){[].forEach.call(document.querySelectorAll(".expandable, .card"),function(el){
  if(el.classList.contains("card")&&el.closest(".expandable"))return;add(el)})}
 document.addEventListener("keydown",function(e){if(e.key==="Escape")close()});
 // Entering is asynchronous: redraw at the final size; leaving by Esc closes.
 document.addEventListener("fullscreenchange",function(){if(!document.fullscreenElement){if(open)close();else redraw()}else redraw()});
 // Scripts redraw their cards; give new cards their button.
 var t=null;new MutationObserver(function(){clearTimeout(t);t=setTimeout(scan,50)}).observe(document.body,{childList:true,subtree:true});
 scan();
})();
