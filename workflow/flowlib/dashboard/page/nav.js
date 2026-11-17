(function(){
 [].forEach.call(document.querySelectorAll("details.sec"),function(d){
  try{var v=sessionStorage.getItem("ftsec-"+d.id);if(v!==null)d.open=v=="1"}catch(e){}
  d.addEventListener("toggle",function(){try{sessionStorage.setItem("ftsec-"+d.id,d.open?"1":"0")}catch(e){}})});
 [].forEach.call(document.querySelectorAll(".nav a"),function(a){
  a.addEventListener("click",function(){var d=document.querySelector(a.getAttribute("href"));if(d&&d.tagName=="DETAILS")d.open=true})});
})();
