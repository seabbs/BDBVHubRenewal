// Full screen for every card and for the nowcasts view with its controls.
// Each gets a button; it opens the Fullscreen API where the browser has it,
// else a fixed overlay; Esc or the button closes it. Plots drawn by script
// redraw on the resize event sent after each change.
(function(){
 var open=null;
 function redraw(){window.dispatchEvent(new Event("resize"))}
 function close(){if(!open)return;var el=open;open=null;
  el.classList.remove("fs");document.body.classList.remove("fsopen");
  var b=el.querySelector(":scope > .expand");if(b){b.textContent="⤢";b.setAttribute("aria-label","Full screen");b.focus()}
  if(document.fullscreenElement&&document.exitFullscreen)document.exitFullscreen().catch(function(){});
  redraw()}
 function show(el){if(open)close();open=el;
  el.classList.add("fs");document.body.classList.add("fsopen");
  var b=el.querySelector(":scope > .expand");if(b){b.textContent="✕";b.setAttribute("aria-label","Close full screen");b.focus()}
  if(el.requestFullscreen)el.requestFullscreen().catch(function(){});
  redraw()}
 function add(el){if(el.querySelector(":scope > .expand"))return;
  var b=document.createElement("button");b.type="button";b.className="expand";
  b.textContent="⤢";b.setAttribute("aria-label","Full screen");b.title="Full screen (Esc closes)";
  b.addEventListener("click",function(){open===el?close():show(el)});
  el.insertBefore(b,el.firstChild)}
 // Cards inside the nowcasts view open with the whole view.
 function scan(){[].forEach.call(document.querySelectorAll(".expandable, .card"),function(el){
  if(el.classList.contains("card")&&el.closest(".expandable"))return;add(el)})}
 document.addEventListener("keydown",function(e){if(e.key==="Escape")close()});
 // Entering is asynchronous: redraw at the final size; leaving by Esc closes.
 document.addEventListener("fullscreenchange",function(){if(!document.fullscreenElement){if(open)close()}else redraw()});
 // Scripts redraw their cards; give new cards their button.
 var t=null;new MutationObserver(function(){clearTimeout(t);t=setTimeout(scan,50)}).observe(document.body,{childList:true,subtree:true});
 scan();
})();
