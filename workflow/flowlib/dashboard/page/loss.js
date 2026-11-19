(function(){
var D=JSON.parse(document.getElementById("ldata").textContent);
var $=function(i){return document.getElementById(i)};
var STY=["","6 3","8 3 2 3","2 3"],SPAL=["#16a34a","#ea580c","#9333ea","#0891b2","#2563eb","#ca8a04","#db2777"];
var PARTS=[["nowcast","nowcast steps","6 3"],["forecast","forecast steps","2 3"]];
var SRCS=[];D.runs.forEach(function(r){Object.keys(r.bysrc).concat(Object.keys(r.src0||{})).forEach(function(k){if(SRCS.indexOf(k)<0)SRCS.push(k)})});SRCS.sort();
var SRC={};SRCS.forEach(function(k,i){SRC[k]=SPAL[i%SPAL.length]});
var LVLS=[];D.runs.forEach(function(r){Object.keys(r.ck_lvl||{}).concat(Object.keys(r.lvl0||{})).forEach(function(k){if(LVLS.indexOf(k)<0)LVLS.push(k)})});
var LVL={};LVLS.forEach(function(k,i){LVL[k]=SPAL[(i+3)%SPAL.length]});
// The held-out loss type shown (Loss type), kept for the session.
var LT="total";try{LT=sessionStorage.getItem("ftlt")||"total"}catch(e){}
document.addEventListener("click",function(e){var b=e.target.closest("[data-lt]");if(!b)return;LT=b.dataset.lt;try{sessionStorage.setItem("ftlt",LT)}catch(x){}draw()});
function sw(c,d){return '<svg width=22 height=8 viewBox="0 0 22 8"><line x1=0 x2=22 y1=4 y2=4 stroke="'+c+'" stroke-width=2'+(d?' stroke-dasharray="'+d+'"':"")+' /></svg>'}
function roll(p,k){var o=[],a=[];p.forEach(function(q){a.push(q[1]);if(a.length>k)a.shift();o.push([q[0],a.reduce(function(x,y){return x+y},0)/a.length])});return o}
function lticks(lo,hi){var o=[];for(var e=Math.floor(lo)-1;e<=Math.ceil(hi);e++)[1,2,3,5,7].forEach(function(m){var v=m*Math.pow(10,e),l=Math.log10(v);if(l>=lo&&l<=hi)o.push(v)});
 while(o.length>6){o=o.filter(function(v,i){return i%2==0})}return o}
function fmtT(v){return +v.toPrecision(2)}
// A point marker: circle (all steps), triangle (nowcast), square (forecast).
function mark(shape,x,y,c,t){var tt='<title>'+t+'</title>';
 if(shape=="^")return '<path d="M'+x+','+(y-4)+'l4,7h-8z" fill="'+c+'">'+tt+'</path>';
 if(shape=="s")return '<rect x="'+(x-3)+'" y="'+(y-3)+'" width=6 height=6 fill="none" stroke="'+c+'" stroke-width=1.5>'+tt+'</rect>';
 return '<circle cx="'+x+'" cy="'+y+'" r=3.2 fill="'+c+'">'+tt+'</circle>'}
function msw(shape){return '<svg width=12 height=10 viewBox="0 0 12 10">'+mark(shape,6,5,"var(--fg)","")+'</svg>'}
// A chart fills its card when the card is full screen (expand.js), else
// draws at 440 x 280 and scales with the card's width.
function size(el){var f=window.ftFull&&window.ftFull(el);if(!f)return [440,280];
 return [Math.max(440,el.clientWidth-8),Math.max(280,f.clientHeight-140)]}
function chart(o,el){
 var wh=size(el),W=wh[0],H=wh[1],L=40,R=8,T=8,B=26,xs=[],ys=[];
 var ser=o.series.map(function(s){var p=s.pts.filter(function(q){return q[1]!=null&&q[1]>0&&(o.xmax==null||q[0]<=o.xmax)});return {label:s.label,color:s.color,kind:s.kind,dash:s.dash,shape:s.shape,pts:p}}).filter(function(s){return s.pts.length});
 if(!ser.length)return '<h3>'+o.title+'</h3><p class=mut>no data for the selected runs</p>';
 ser.forEach(function(s){s.pts.forEach(function(q){xs.push(q[0]);ys.push(Math.log10(q[1]))})});
 var x1=o.xmax!=null?o.xmax:Math.max.apply(null,xs.concat([1])),y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);
 if(o.hl){y0=Math.min(y0,0);y1=Math.max(y1,0)}
 var pd=(y1-y0)*0.06||0.1;y0-=pd;y1+=pd;
 var sx=function(x){return L+x/x1*(W-L-R)},sy=function(v){return T+(1-(Math.log10(v)-y0)/(y1-y0))*(H-T-B)};
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%"'+(window.ftFull&&window.ftFull(el)?' height="'+H+'"':'')+'>';
 lticks(y0,y1).forEach(function(v){h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-4)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+fmtT(v)+'</text>'});
 var st=Math.pow(10,Math.floor(Math.log10(x1/4)));st=[1,2,2.5,5,10].map(function(m){return m*st}).find(function(v){return v>=x1/5});
 for(var x=0;x<=x1;x+=st)h+='<text x="'+sx(x)+'" y="'+(H-B+12)+'" text-anchor=middle class=ax>'+(+x.toPrecision(6))+'</text>';
 h+='<text x="'+(W/2)+'" y="'+(H-2)+'" text-anchor=middle class=ax>simulations trained</text>';
 if(o.hl)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(1)+'" y2="'+sy(1)+'" class=mark />';
 (D.marks||[]).forEach(function(m){if(m[0]<=x1)h+='<line x1="'+sx(m[0])+'" x2="'+sx(m[0])+'" y1="'+T+'" y2="'+(H-B)+'" class=mark /><text x="'+(sx(m[0])+2)+'" y="'+(T+9)+'" class=ax>'+m[1]+'</text>'});
 ser.forEach(function(s){var d=s.dash?' stroke-dasharray="'+s.dash+'"':"";
  if(s.kind=="marks"){s.pts.forEach(function(q){h+=mark(s.shape,sx(q[0]),sy(q[1]),s.color,s.label+' at '+Math.round(q[0])+' simulations: '+(+q[1].toPrecision(4)))});return}
  if(s.kind=="dots")s.pts.forEach(function(q){h+='<circle cx="'+sx(q[0])+'" cy="'+sy(q[1])+'" r=1.8 fill="'+s.color+'" opacity=".35" />'});
  else{var pl=s.pts.map(function(q){return sx(q[0])+","+sy(q[1])}).join(" ");
   h+='<polyline points="'+pl+'" fill=none stroke="'+s.color+'" stroke-width=1.6'+d+(s.kind=="faint"?' opacity=".4"':'')+' />';
   if(s.kind=="pts")s.pts.forEach(function(q){h+='<circle cx="'+sx(q[0])+'" cy="'+sy(q[1])+'" r=2.2 fill="'+s.color+'"><title>'+s.label+' step '+q[0]+': '+(+q[1].toPrecision(4))+'</title></circle>'})}});
 var leg=o.leg||ser.map(function(s){return '<span><i style="background:'+s.color+'"></i>'+s.label+(s.dash?" (dashed)":"")+'</span>'}).join("");
 return '<h3>'+o.title+ftInfo(o.info)+'</h3><div class=leg>'+leg+'</div>'+h+'</svg>'}
// The three cards persist across redraws, so a full-screen card stays open
// and its expand button (expand.js) is kept.
function cards(list){var box=$("lossplots");
 while(box.children.length<list.length){var c=document.createElement("div");c.className="card";box.appendChild(c)}
 list.forEach(function(o,i){var c=box.children[i],b=c.querySelector(":scope > .expand");
  c.innerHTML=chart(o,c);if(b)c.insertBefore(b,c.firstChild)})}
function draw(){
 var mode=window.ftMode||"start",rng=window.ftRng||"full",on=ftSel.get(),rel=mode!="raw";
 var xmax=rng=="zoom"?D.xlim:null,runs=D.runs.filter(function(r){return on.indexOf(r.key)>=0});
 var suf={raw:"",start:", relative to start",base:", relative to base Chronos-2"}[mode],hl=rel;
 var tr=[],held=[],byRuns=[];
 runs.forEach(function(r){var i=D.runs.indexOf(r);
  var p=r.train;if(rel&&r.tr0)p=p.map(function(q){return [q[0],q[1]/r.tr0]});
  if(r.live)tr.push({color:r.color,pts:p,kind:"dots"});
  tr.push({label:r.label,color:r.color,pts:roll(p,20),kind:r.live?"line":"faint"});
  // The held-out loss of the chosen type (Loss type): the subset monitor
  // at each evaluation as a line, the full set at step 0 and at each
  // checkpoint as points. The subset has no step-0 value, so relative to
  // start is to its first evaluation; it has no base Chronos-2 value.
  function scaled(pts,d){return d?pts.map(function(q){return [q[0],q[1]/d]}):[]}
  function add(lab,colour,dash,sub,z,bz,ck,shape,key){
   if(sub&&sub.length&&mode!="base"){var s=mode=="start"?scaled(sub,sub[0][1]):sub;
    held.push({label:r.label+" "+lab+" subset",color:colour,pts:s,kind:"line",dash:dash,sub:1,key:key})}
   var f=(z!=null?[[0,z]]:[]).concat(ck||[]);if(!f.length)return;
   if(rel)f=scaled(f,mode=="start"?z:bz);
   if(f.length)held.push({label:r.label+" "+lab+" full set",color:colour,pts:f,kind:"marks",shape:shape,key:key})}
  var dash=STY[i%4];
  if(LT=="total")add("all steps",r.color,"",r.held,r.full0,r.bfull,r.ck,"o");
  else if(LT=="nowcast"||LT=="forecast")add(LT+" steps",r.color,"",(r.held_parts||{})[LT],(r.parts0||{})[LT],(r.bparts0||{})[LT],(r.ck_parts||{})[LT],"o");
  else if(LT=="layout")SRCS.forEach(function(s){add(s,SRC[s],dash,(r.bysrc||{})[s],(r.src0||{})[s],(r.bsrc||{})[s],(r.ck_src||{})[s],"o",s);if(held.some(function(h){return h.key==s})&&byRuns.indexOf(r)<0)byRuns.push(r)});
  else LVLS.forEach(function(l){add(l,LVL[l],dash,null,(r.lvl0||{})[l],(r.blvl0||{})[l],(r.ck_lvl||{})[l],"o",l);if(held.some(function(h){return h.key==l})&&byRuns.indexOf(r)<0)byRuns.push(r)})});
 // Run colours are in the shared selector at the top; each legend names
 // only its own encodings. By layout or level, colour is the layout or
 // level and line style the run.
 function has(f){return held.some(f)}
 var keyed=LT=="layout"?SRCS.map(function(s){return [s,SRC[s]]}):LT=="level"?LVLS.map(function(l){return [l,LVL[l]]}):[];
 var heldLeg=keyed.filter(function(k){return has(function(h){return h.key==k[0]})}).map(function(k){return '<span><i style="background:'+k[1]+'"></i>'+k[0]+'</span>'}).join("")
  +(keyed.length?byRuns.map(function(r){return '<span>'+sw("var(--fg)",STY[D.runs.indexOf(r)%4])+' '+r.label+'</span>'}).join(""):"")
  +(has(function(h){return h.sub})?'<span>'+sw("var(--fg)","")+' subset (first 100 blocks), each evaluation</span>':"")
  +(has(function(h){return !h.sub})?'<span>'+msw("o")+' full set, step 0 and each checkpoint</span>':"")
  +(LT=="level"?'<span class=mut>by level: step 0 and checkpoints only (the trainer logs no level loss in between)</span>':"")
  +(mode=="base"&&runs.some(function(r){return r.held.length})?'<span class=mut>subset: no base Chronos-2 value</span>':"")
  +(mode=="start"&&has(function(h){return h.sub})?'<span class=mut>subset: relative to its first evaluation</span>':"");
 var trLeg=(runs.some(function(r){return r.live})?'<span><svg width=12 height=10><circle cx=6 cy=5 r=1.8 fill="var(--fg)" opacity=".35"/></svg> logged values (live run)</span>':"")+'<span>'+sw("var(--fg)","")+' rolling mean of 20</span>';
 var ttl="Held-out loss, "+{total:"all steps",nowcast:"nowcast steps",forecast:"forecast steps",layout:"by layout",level:"by level"}[LT]+", subset and full set";
 cards([{title:"Training loss"+suf,info:"train",series:tr,xmax:xmax,hl:hl,leg:trLeg},{title:ttl+suf,info:"held",series:held,xmax:xmax,hl:hl,leg:heldLeg}]);
 [].forEach.call(document.querySelectorAll("[data-lt]"),function(b){b.classList.toggle("on",b.dataset.lt==LT)});
 var refs=ftSel.refsOn();
 $("lossrefs").innerHTML=refs.map(function(r){return '<span class="tog dis" title="'+r.label+' is a reference with no training or held-out loss">'+r.label+'</span>'}).join("");
}
window.addEventListener("ftsel",draw);
// Full screen and window size changes redraw at the new size, same state.
var rsz=null;window.addEventListener("resize",function(){clearTimeout(rsz);rsz=setTimeout(draw,120)});
// New data from the server (live.js): redraw with the same state.
window.addEventListener("ftdata",function(e){if(e.detail!=="loss")return;D=JSON.parse($("ldata").textContent);draw()});window.addEventListener("ftmode",draw);
draw();
})();
