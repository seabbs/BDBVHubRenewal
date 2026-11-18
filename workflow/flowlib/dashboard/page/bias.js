(function(){
var D=JSON.parse(document.getElementById("bdata").textContent);
var $=function(i){return document.getElementById(i)};
var PAIRS=D.pairs,COL=D.col,sel0=null;
function nice(lo,hi){var st=(hi-lo)/4,m=Math.pow(10,Math.floor(Math.log10(st))),k=[1,2,2.5,5,10].find(function(x){return x*m>=st})*m,o=[];
 for(var v=Math.ceil(lo/k)*k;v<=hi+1e-9;v+=k)o.push(+v.toFixed(6));return o}
function heat(rows,title,lo,hi){var div=lo<0&&hi>0,amp=Math.max(Math.abs(lo),Math.abs(hi))||1,nr=rows.length,nc=rows[0].length,cw=140/Math.max(nr,nc);
 var h='<svg viewBox="0 0 140 140" width="140" height="140">';
 rows.forEach(function(r,i){r.forEach(function(v,j){var col;
  if(div){var a=Math.abs(v)/amp;col=v>0?"rgba(220,38,38,"+a.toFixed(2)+")":"rgba(37,99,235,"+a.toFixed(2)+")"}
  else{var a2=(v-lo)/((hi-lo)||1);col="rgba(37,99,235,"+(0.08+0.92*a2).toFixed(2)+")"}
  h+='<rect x="'+(j*cw).toFixed(1)+'" y="'+(i*cw).toFixed(1)+'" width="'+(cw-0.5).toFixed(1)+'" height="'+(cw-0.5).toFixed(1)+'" fill="'+col+'"><title>'+title+' layer '+i+' head '+j+': '+(+v.toPrecision(4))+'</title></rect>'})});
 return h+'</svg>'}
function lineplot(run){
 var W=480,H=280,L=44,R=10,T=8,B=28,all=[];
 PAIRS.forEach(function(k){(run.pts[k.key]||[]).forEach(function(p){all.push(p)})});
 if(!all.length)return "";
 var xs=all.map(function(p){return p[0]}),x0=Math.min.apply(null,xs),x1=Math.max(Math.max.apply(null,xs),x0+1);
 var ys=all.map(function(p){return p[1]}),y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys),pd=(y1-y0)*0.25||0.01;y0-=pd;y1+=pd;
 var sx=function(x){return L+(x-x0)/(x1-x0)*(W-L-R)},sy=function(y){return T+(1-(y-y0)/(y1-y0))*(H-T-B)};
 var s='<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'">';
 nice(y0,y1).forEach(function(v){if(v>=y0&&v<=y1)s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-5)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+(+v.toPrecision(3))+'</text>'});
 nice(x0,x1).forEach(function(v){if(v>=x0&&v<=x1)s+='<text x="'+sx(v)+'" y="'+(H-B+13)+'" text-anchor=middle class=ax>'+v+'</text>'});
 s+='<text x="'+(L+W-R)/2+'" y="'+(H-3)+'" text-anchor=middle class=ax>step</text>';
 if(y0<0&&y1>0)s+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(0)+'" y2="'+sy(0)+'" class=mark />';
 PAIRS.forEach(function(k,n){var pts=run.pts[k.key]||[];if(!pts.length)return;var c=COL[k.key];
  var up=pts.map(function(p){return sx(p[0])+","+sy(p[3])}).join(" "),dn=pts.slice().reverse().map(function(p){return sx(p[0])+","+sy(p[2])}).join(" "),mean=pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ");
  s+='<g class=rel><polygon class=band points="'+up+" "+dn+'" fill="'+c+'" /><polyline points="'+mean+'" fill=none stroke="'+c+'" stroke-width=2 /><polyline points="'+mean+'" fill=none stroke="transparent" stroke-width=12 />';
  pts.forEach(function(p){s+='<circle cx="'+sx(p[0])+'" cy="'+sy(p[1])+'" r=2.6 fill="'+c+'"><title>'+k.label+' step '+p[0]+': mean '+(+p[1].toPrecision(4))+', range '+(+p[2].toPrecision(4))+' to '+(+p[3].toPrecision(4))+'</title></circle>'});
  s+='<text x="'+(L+6)+'" y="'+(T+12+n*12)+'" class=ax style="fill:'+c+'">'+k.label+'</text></g>'});
 return s+'</svg>'}
function draw(){
 var on=ftSel.get(),runs=D.runs.filter(function(r){return on.indexOf(r.key)>=0});
 var el=$("biasdyn");
 $("biasrefs").innerHTML=ftSel.refsOn().map(function(r){return '<span class="tog dis" title="'+r.label+' is a reference with no attention bias">'+r.label+'</span>'}).join("");
 var lr=document.getElementById("lrcard");if(lr)lr.style.display=on.indexOf(lr.dataset.run)>=0?"":"none";
 if(!runs.length){el.innerHTML='<p class=mut style="line-height:1.4">none of the selected runs has an attention bias module</p>';return}
 if(!sel0||!runs.some(function(r){return r.key==sel0}))sel0=runs[0].key;
 var run=runs.find(function(r){return r.key==sel0});
 var lat=run.latest||{},keys=PAIRS.filter(function(k){return lat[k.key]}),flat=[];
 keys.forEach(function(k){lat[k.key].forEach(function(r){r.forEach(function(v){flat.push(v)})})});
 var lo=flat.length?Math.min.apply(null,flat):0,hi=flat.length?Math.max.apply(null,flat):1;
 var tiles=keys.map(function(k){return '<div class=tile4><div class=mut title="'+k.label+'">'+k.key+'</div>'+heat(lat[k.key],k.key,lo,hi)+'</div>'}).join("");
 var tabs=runs.length>1?'<div class=sctl style="margin:0 0 4px">'+runs.map(function(r){return '<button class="tog'+(r.key==sel0?" on":"")+'" data-b="'+r.key+'" style="border-left:4px solid '+r.color+'">'+r.label+'</button>'}).join("")+'</div>':"";
 el.innerHTML=(tabs?'<div class=biastabs>'+tabs+'</div>':"")+'<div class="card biasline"><div class=cardinfo>'+ftInfo("biasline")+'</div>'+lineplot(run)+'</div>'+(tiles?'<div><div class=tiles4>'+tiles+'</div><div class=mut>'+ftInfo("biasheat")+ftInfo("relations")+' step '+lat.step+', colour scale '+(+lo.toPrecision(3))+' to '+(+hi.toPrecision(3))+', shared; rows layers, columns heads</div></div>':"");
 var tb=el.querySelector(".sctl");if(tb)tb.onclick=function(e){var b=e.target.closest("button");if(b){sel0=b.dataset.b;draw()}}}
window.addEventListener("ftsel",draw);
// New data from the server (live.js): redraw with the same state.
window.addEventListener("ftdata",function(e){if(e.detail!=="bias")return;D=JSON.parse($("bdata").textContent);draw()});
draw();
})();
