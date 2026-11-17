(function(){var m="start",r="zoom";
 try{m=sessionStorage.getItem("ftmode")||m;r=sessionStorage.getItem("ftrng")||r}catch(e){}
 function paint(){var c=document.body.className.replace(/\b[mr]-\w+/g,"").trim();
  document.body.className=c+" m-"+m+" r-"+r;
  [].forEach.call(document.querySelectorAll(".modebar button"),function(b){
   b.classList.toggle("on",b.dataset.mode==m||b.dataset.rng==r)});
  try{sessionStorage.setItem("ftmode",m);sessionStorage.setItem("ftrng",r)}catch(e){}
  window.ftMode=m;window.ftRng=r;window.dispatchEvent(new Event("ftmode"))}
 [].forEach.call(document.querySelectorAll(".modebar"),function(bar){
  bar.onclick=function(e){var b=e.target.closest("button");if(!b)return;
   if(b.dataset.mode)m=b.dataset.mode;if(b.dataset.rng)r=b.dataset.rng;paint()}});
 paint()})();
