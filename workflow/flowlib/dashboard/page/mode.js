(function(){var m="start";
 try{m=sessionStorage.getItem("ftmode")||m}catch(e){}
 function set(v){m=v;document.body.className=document.body.className.replace(/\bm-\w+/g,"")+" m-"+v;
  window.ftMode=v;
  [].forEach.call(document.querySelectorAll(".modebar button"),function(b){b.classList.toggle("on",b.dataset.mode==v)});
  try{sessionStorage.setItem("ftmode",v)}catch(e){}
  window.dispatchEvent(new Event("ftmode"))}
 [].forEach.call(document.querySelectorAll(".modebar"),function(bar){
  bar.onclick=function(e){var b=e.target.closest("button");if(b)set(b.dataset.mode)}});
 set(m)})();
