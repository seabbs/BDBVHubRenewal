(function(){
var P=JSON.parse(document.getElementById("cdata").textContent);
var $=function(i){return document.getElementById(i)};
var SRCS=[["all","all"],["weekly","weekly"],["daily","daily"],["3day","3-day"]];
var S={src:"all"};
try{var sv=JSON.parse(sessionStorage.getItem("ftck")||"null");if(sv)S=sv}catch(e){}
function fmt(v){return (+v).toPrecision(4)}
function draw(){
 var mode=window.ftMode||"start";
 $("cksrc").innerHTML='<span class=mut>Source</span>'+SRCS.map(function(a){return '<button class="tog'+(S.src==a[0]?" on":"")+'" data-s="'+a[0]+'">'+a[1]+'</button>'}).join("");
 var rel=mode!="raw",bad=rel&&S.src!="all";
 var runs=P.runs.map(function(r){
  var den=mode=="start"?r.full0:r.bfull,pts=[];
  if(!rel||den){
   if(S.src=="all"&&r.full0)pts.push({n:0,v:rel?r.full0/den:r.full0,t:"start (step 0): "+fmt(r.full0)});
   r.points.forEach(function(p){var v=p[S.src];if(v==null)return;
    pts.push({n:p.n,v:rel?v/den:v,t:r.label+" n"+p.n+" "+p.hash+", step "+p.step+": "+fmt(v)+(rel?" ("+fmt(v/den)+" relative)":"")+(p.real!=null?"; real-validation "+fmt(p.real):"")})})}
  return {r:r,pts:pts}});
 var all=[].concat.apply([],runs.map(function(x){return x.pts}));
 if(bad||!all.length){$("ckplot").innerHTML='<p class=mut>'+(bad?"Per-source start losses are not recorded on the full held-out set, so relative values are only available for all sources. Choose raw.":"No scored checkpoints yet.")+'</p>';return}
 var ns=Array.from(new Set(all.filter(function(p){return p.n>0}).map(function(p){return p.n}))).sort(function(a,b){return a-b});
 var zx=ns[0]/2,W=640,H=320,L=48,R=16,T=12,B=40;
 var x0=Math.log2(zx),x1=Math.log2(ns[ns.length-1]*1.15);
 var sx=function(n){return L+(Math.log2(n||zx)-x0)/(x1-x0)*(W-L-R)};
 var vs=all.map(function(p){return p.v});if(rel)vs.push(1);
 var lo=Math.min.apply(null,vs),hi=Math.max.apply(null,vs),pd=(hi-lo||0.05)*0.1;lo-=pd;hi+=pd;
 var sy=function(v){return T+(1-(v-lo)/(hi-lo))*(H-T-B)};
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var st=(hi-lo)/4,m=Math.pow(10,Math.floor(Math.log10(st))),k=[1,2,2.5,5,10].find(function(x){return x*m>=st})*m;
 for(var v=Math.ceil(lo/k)*k;v<=hi;v+=k)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-5)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+(+v.toFixed(4))+'</text>';
 [[zx,"start"]].concat(ns.map(function(n){return [n,""+n]})).forEach(function(t){h+='<line x1="'+sx(t[0])+'" x2="'+sx(t[0])+'" y1="'+T+'" y2="'+(H-B)+'" class=grid /><text x="'+sx(t[0])+'" y="'+(H-B+14)+'" text-anchor=middle class=ax>'+t[1]+'</text>'});
 h+='<text x="'+(W/2)+'" y="'+(H-6)+'" text-anchor=middle class=ax>simulations (log2)</text>';
 if(rel&&1>=lo&&1<=hi)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(1)+'" y2="'+sy(1)+'" class=mark />';
 runs.forEach(function(x){if(!x.pts.length)return;var c=x.r.color;
  h+='<polyline points="'+x.pts.map(function(p){return sx(p.n)+","+sy(p.v)}).join(" ")+'" fill=none stroke="'+c+'" stroke-width=1.8 />';
  x.pts.forEach(function(p){h+='<circle cx="'+sx(p.n)+'" cy="'+sy(p.v)+'" r=3.4 fill="'+c+'"><title>'+p.t+'</title></circle>';
   h+='<text x="'+(sx(p.n)+5)+'" y="'+(sy(p.v)-5)+'" class=ax>'+(+p.v.toPrecision(3))+'</text>'})});
 h+='</svg>';
 var leg=runs.map(function(x){return '<span><i style="background:'+x.r.color+'"></i>'+x.r.label+'</span>'}).join("");
 $("ckplot").innerHTML=h+'<div class=leg>'+leg+'</div><p class=mut>Held-out loss on the full held-out set at each saved checkpoint'+(rel?', '+(mode=="start"?"divided by the run\'s own step-0 loss":"divided by the base Chronos-2 loss on the same blocks")+'. Reference line is 1':'')+'. Hover for hash, step and real-validation loss.</p>';
}
$("cksrc").onclick=function(e){var b=e.target.closest("button");if(!b)return;S.src=b.dataset.s;try{sessionStorage.setItem("ftck",JSON.stringify(S))}catch(x){}draw()};
window.addEventListener("ftmode",draw);
draw();
})();
