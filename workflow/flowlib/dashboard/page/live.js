// Update the page in place every 60 s, keeping every selection, zoom, open
// section and full-screen card. The server sends each part's hash and the
// html of the parts whose hash changed (/sections.json?have=key:hash,...).
// A part with data (<script type=application/json>) gets only its new data
// and an ftdata event, so its script redraws with its own state; a part
// without is replaced, except while one of its cards is full screen.
(function(){
 var have={},first=true;
 function apply(key,html){
  if(key==="bar"){var b=document.getElementById("livebar");if(b)b.innerHTML=html;return true}
  var sec=document.getElementById(key);if(!sec)return true;
  var t=document.createElement("template");t.innerHTML=html;
  var data=t.content.querySelectorAll('script[type="application/json"]');
  if(data.length){
   [].forEach.call(data,function(s){var o=document.getElementById(s.id);if(o)o.textContent=s.textContent});
   var lr=t.content.querySelector("#lrcard"),old=document.getElementById("lrcard");
   if(lr&&old&&!old.classList.contains("fs"))old.replaceWith(lr);
   window.dispatchEvent(new CustomEvent("ftdata",{detail:key}));return true}
  if(sec.querySelector(".fs"))return false;
  [].slice.call(sec.childNodes).forEach(function(n){if(n.nodeName!=="SUMMARY")sec.removeChild(n)});
  sec.appendChild(t.content);return true}
 function poll(){
  if(window.ftPlaying)return;
  var q=Object.keys(have).map(function(k){return k+":"+have[k]}).join(",");
  fetch("/sections.json?have="+encodeURIComponent(q)).then(function(r){return r.json()}).then(function(j){
   // The first answer only records what the page was built with.
   Object.keys(j).forEach(function(k){var v=j[k];
    if(first||v.html===undefined){have[k]=v.hash;return}
    if(apply(k,v.html))have[k]=v.hash});
   first=false}).catch(function(){})}
 poll();setInterval(poll,60000);
})();
