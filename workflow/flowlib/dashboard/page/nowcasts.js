(function(){
var D=JSON.parse(document.getElementById("gdata").textContent);
var DSS=D.datasets,RUNS=D.runs;
var $=function(i){return document.getElementById(i)};
var DASH=["","6 3","8 3 2 3","2 3"],MK=["circle","square","triangle","diamond"];
var DEF=D.defaults||{};
var FIRST=(DSS.find(function(d){return !d.empty})||DSS[0]).key;
var S={ds:FIRST,locs:[],src:DEF.source||"",target:null,oi:null,latest:true,nd:1,step:2,on:null,chk:{},yfit:0,ylog:false,bands:0};
var KEY="ftg5";
try{var sv=JSON.parse(sessionStorage.getItem(KEY)||"null");if(sv)Object.keys(sv).forEach(function(k){S[k]=sv[k]})}catch(e){}
function cur(){return DSS.find(function(d){return d.key==S.ds})||DSS[0]}
// A dataset with no nowcasts yet (held-out simulations until they are scored) shows why.
function empty(){$("gdate").textContent="";$("gck").innerHTML="";$("glocs").innerHTML="";$("gsum").textContent="Locations";
 $("gplot").innerHTML='<p class=mut>No nowcasts for '+C.label+' yet'+(C.missing?': '+C.missing:'')+'.</p>';$("gwis").innerHTML='';save()}
var C=cur();S.ds=C.key;
function ofRun(k){return C.meta.filter(function(m){return m.run==k}).sort(function(a,b){return a.n-b.n})}
function refs(){return C.meta.filter(function(m){return m.run=="ref"})}
function ORI(){return C.origins}
function TGT(){var a=C.targets||[];return a.indexOf(S.target)>=0?S.target:(a.indexOf(DEF.target)>=0?DEF.target:a[0]||"")}
function tlab(v){return v?v.replace(/^(\d+)day$/,"$1-day"):"all"}
S.locs=(S.locs||[]).filter(function(l){return C.locs.indexOf(l)>=0});if(!S.locs.length&&C.locs.length)S.locs=[C.locs[0]];
if(S.latest||ORI().indexOf(S.origin)<0){S.oi=ORI().length-1;S.latest=true}else S.oi=ORI().indexOf(S.origin);
// Series ids are the configured reference keys and checkpoint hashes.
function onKey(id){return id}
function isOn(k){return ftSel.get().indexOf(k)>=0}
function save(){try{sessionStorage.setItem(KEY,JSON.stringify({ds:S.ds,locs:S.locs,src:S.src,target:S.target,latest:S.latest,origin:ORI()[S.oi],nd:S.nd,step:S.step,chk:S.chk}))}catch(e){}}
// A run's chosen checkpoints, smallest first, so lines, legend and scores share one order.
function chosen(k){var all=ofRun(k),ids=all.map(function(m){return m.id}).filter(function(i){return (S.chk[k]||[]).indexOf(i)>=0});
 if(!ids.length&&all.length)ids=[all[all.length-1].id];return ids}
function effective(){var o=[];
 RUNS.forEach(function(r){if(isOn(r.key)){var ids=chosen(r.key);
  ids.forEach(function(id,i){o.push({id:id,run:r.key,rank:ids.length-1-i})})}});
 refs().forEach(function(m){if(isOn(onKey(m.id)))o.push({id:m.id,run:"ref",rank:0})});
 return o}
function ms(s){return Date.parse(s+"T00:00:00Z")}
function dates(){var O=ORI(),out=[];
 for(var k=0;k<S.nd;k++){var tg=ms(O[S.oi])-k*S.step*7*864e5,j=-1;
  for(var i=0;i<O.length;i++){if(ms(O[i])<=tg)j=i}
  if(j>=0&&out.indexOf(O[j])<0)out.unshift(O[j])}
 return out}
var cache={};
function load(loc){var pre=C.preload||{};
 if(pre[loc])return Promise.resolve(pre[loc]);
 var k=C.key+"/"+loc;if(cache[k])return Promise.resolve(cache[k]);
 return fetch("/data/"+encodeURIComponent(C.key)+"/"+encodeURIComponent(loc)+".json").then(function(r){return r.json()})
  .then(function(j){cache[k]=j;return j})}
function colour(id){var m=C.meta.find(function(x){return x.id==id});return m?m.color:"#64748b"}
function id2lab(id){var m=C.meta.find(function(x){return x.id==id});return m?m.label:id}
function fmt(v){return v>=1000?Math.round(v).toLocaleString("en-GB"):v==0?"0":(+v).toPrecision(3)}
function dashAttr(rank){var d=DASH[rank%4];return d?' stroke-dasharray="'+d+'"':""}
function mark(k,x,y,c,t,r){r=r||3;var s="";
 if(k=="circle")s='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+c+'">';
 else if(k=="square")s='<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+c+'">';
 else if(k=="triangle")s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+(y+r)+' '+(x-r-1)+','+(y+r)+'" fill="'+c+'">';
 else s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+y+' '+x+','+(y+r+1)+' '+(x-r-1)+','+y+'" fill="'+c+'">';
 return s+'<title>'+t+'</title>'+(k=="circle"?"</circle>":k=="square"?"</rect>":"</polygon>")}
var CLIPN=0;
function panel(d,loc,ds,eff,small,W,FH){
 var HI=[6,4,3][S.yfit||0],LO=6-HI,H=Math.round(FH||(small?Math.max(170,W*0.62):Math.min(400,Math.max(260,W*0.5)))),L=46,R=8,T=8,B=24,ymax=1,i;
 var t0=ms(ds[0])-70*864e5,t1=ms(ds[ds.length-1])+35*864e5;
 function within(l){var o=[];l=l||{t:[],y:[]};for(i=0;i<l.t.length;i++){var x=ms(l.t[i]);if(x>=t0&&x<=t1){o.push([x,l.y[i]]);ymax=Math.max(ymax,l.y[i])}}return o}
 var pts=within(d.truth[TGT()]),lat=within((d.latest||{})[TGT()]);
 var fs=[];
 eff.forEach(function(e){ds.forEach(function(o,di){var s=((d.series[e.id]||{})[TGT()]||{})[S.src],f=s&&s[o];
  if(f){fs.push({e:e,f:f,di:di,o:o});f.q.forEach(function(q){ymax=Math.max(ymax,q[HI])})}})});
 var runBands=eff.some(function(e){return e.run!="ref"&&fs.some(function(z){return z.e===e})});
 ymax*=1.06;
 // y from the data shown, not from zero
 var ylo=Infinity;pts.concat(lat).forEach(function(p){ylo=Math.min(ylo,p[1])});fs.forEach(function(z){z.f.q.forEach(function(q){ylo=Math.min(ylo,q[LO])})});
 var ymin=isFinite(ylo)?Math.max(0,ylo-0.15*(ymax-ylo)):0;
 if(S.ylog)ymin=Math.max(isFinite(ylo)?ylo/1.15:1,0.5);
 var tf=S.ylog?Math.log:function(y){return y},f0=tf(ymin),f1=tf(ymax);
 var sx=function(x){return L+(x-t0)/(t1-t0)*(W-L-R)},sy=function(y){return T+(1-(tf(Math.max(y,S.ylog?ymin/10:-Infinity))-f0)/(f1-f0))*(H-T-B)};
 var cid="cp"+(++CLIPN);
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'"><defs><clipPath id="'+cid+'"><rect x="'+L+'" y="'+T+'" width="'+(W-L-R)+'" height="'+(H-T-B)+'" /></clipPath></defs>';
 var yt=[];
 if(S.ylog){for(var e10=Math.floor(Math.log10(ymin));e10<=Math.ceil(Math.log10(ymax));e10++)[1,2,5].forEach(function(k){var v=k*Math.pow(10,e10);if(v>=ymin&&v<=ymax)yt.push(v)})}
 if(yt.length<3){yt=[];var step=Math.pow(10,Math.floor(Math.log10((ymax-ymin)/4)));
  var m=[1,2,2.5,5,10].find(function(k){return k*step>=(ymax-ymin)/(small?3:5)})*step;
  for(var v=Math.ceil(ymin/m)*m;v<=ymax;v+=m)yt.push(v)}
 yt.forEach(function(v){h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-4)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+fmt(v)+'</text>'});
 var xs=[7,14,28,56,112].find(function(k){return (t1-t0)/(k*864e5)<=W/64})||112;
 for(var x=t0;x<=t1;x+=xs*864e5)h+='<text x="'+sx(x)+'" y="'+(H-B+12)+'" text-anchor=middle class=ax>'+new Date(x).toISOString().slice(5,10)+'</text>';
 h+='<g clip-path="url(#'+cid+')">';
 ds.forEach(function(o){h+='<line x1="'+sx(ms(o))+'" x2="'+sx(ms(o))+'" y1="'+T+'" y2="'+(H-B)+'" class=mark />'});
 var PP=function(f,k){return f.t.map(function(d,j){return sx(ms(d))+","+sy(f.q[j][k])})};
 if(S.bands<2)fs.forEach(function(z){var e=z.e,f=z.f,c=colour(e.id);
  if(e.rank==0&&(e.run!="ref"||!runBands)){var op=ds.length>1?.5:1;
   if(S.bands==0)h+='<polygon points="'+PP(f,0).concat(PP(f,6).reverse()).join(" ")+'" fill="'+c+'" opacity="'+.08*op+'" />';
   h+='<polygon points="'+PP(f,2).concat(PP(f,4).reverse()).join(" ")+'" fill="'+c+'" opacity="'+.16*op+'" />'}});
 fs.forEach(function(z){var e=z.e,f=z.f,c=colour(e.id);
  var P=function(k){return PP(f,k)};
  h+='<polyline points="'+P(3).join(" ")+'" fill=none stroke="var(--bg)" stroke-width=4 opacity=.85 />';
  h+='<polyline points="'+P(3).join(" ")+'" fill=none stroke="'+c+'" stroke-width=2.2'+dashAttr(e.run=="ref"?0:e.rank)+' />';
  f.t.forEach(function(d,j){h+=mark(MK[z.di%4],sx(ms(d)),sy(f.q[j][3]),c,id2lab(e.id)+' (nowcast '+z.o+') '+d+' median '+fmt(f.q[j][3])+' (50%: '+fmt(f.q[j][2])+' to '+fmt(f.q[j][4])+'; '+(f.band||95)+'%: '+fmt(f.q[j][0])+' to '+fmt(f.q[j][6])+')',small?2.2:2.8)})});
 if(lat.length)h+='<polyline points="'+lat.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="var(--fg)" stroke-width=1 stroke-dasharray="3 3" opacity=".6"><title>latest release</title></polyline>';
 if(pts.length){h+='<polyline points="'+pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="var(--fg)" stroke-width=1.4 />';
  pts.forEach(function(p){h+='<circle cx="'+sx(p[0])+'" cy="'+sy(p[1])+'" r=1.8 fill="var(--fg)"><title>truth '+new Date(p[0]).toISOString().slice(0,10)+': '+fmt(p[1])+'</title></circle>'})}
 return {svg:h+'</g></svg>',miss:eff.filter(function(e){return !fs.some(function(z){return z.e===e})}).map(function(e){return id2lab(e.id)})};
}
function legend(eff,ds){
 function sw(c,rank){return '<svg width=26 height=8 viewBox="0 0 26 8"><line x1=0 x2=26 y1=4 y2=4 stroke="'+c+'" stroke-width=2'+dashAttr(rank)+' /></svg>'}
 var leg='<div class=lg><b>truth</b> '+sw("var(--fg)",0)+'</div>';
 if(S.locs.some(function(l){var c=cache[C.key+"/"+l]||(C.preload||{})[l];return c&&c.latest&&c.latest[TGT()]}))leg+='<div class=lg><b>latest release</b> <svg width=26 height=8 viewBox="0 0 26 8"><line x1=0 x2=26 y1=4 y2=4 stroke="var(--fg)" stroke-width=1 stroke-dasharray="3 3" /></svg></div>';
 RUNS.forEach(function(r){var g=eff.filter(function(e){return e.run==r.key});if(!g.length)return;
  leg+='<div class=lg><b>'+r.label+'</b> '+g.map(function(e){return sw(colour(e.id),e.rank)+"n"+C.meta.find(function(x){return x.id==e.id}).n}).join(" ")+'</div>'});
 var rf=eff.filter(function(e){return e.run=="ref"});
 if(rf.length)leg+='<div class=lg><b>reference</b> '+rf.map(function(e){return sw(colour(e.id),0)+id2lab(e.id)}).join(" ")+'</div>';
 if(ds.length>1)leg+='<div class=lg><b>nowcast date</b> '+ds.map(function(o,i){return '<svg width=12 height=12 viewBox="0 0 12 12">'+mark(MK[i%4],6,6,"var(--fg)","",3)+'</svg>'+o}).join(" ")+'</div>';
 return '<div class=leg2>'+leg+'</div>'}
function plots(){
 if(C.empty)return;
 var eff=effective(),ds=dates(),small=S.locs.length>1;
 $("gwrap").className=small?"gwrap wide":"gwrap";
 Promise.all(S.locs.map(load)).then(function(all){
  var miss={},h='<div class=panels style="grid-template-columns:repeat('+(small?"auto-fit,minmax(260px,1fr)":"1,1fr")+')">';
  all.forEach(function(d,i){h+='<div>'+(small?'<h4>'+S.locs[i]+'</h4>':"")+'<div class=gp></div></div>'});
  h+='</div>'+legend(eff,ds)+'<p class=mut>Truth: '+C.truth+'.</p>';
  $("gplot").innerHTML=h;
  // draw each panel at its rendered width so the SVG maps 1:1 to pixels
  // Full screen (expand.js): the panels share the view's height.
  var fs=window.ftFull&&window.ftFull($("gplot")),rows=small?Math.ceil(S.locs.length/Math.max(1,Math.floor($("gplot").clientWidth/260))):1;
  var FH=fs?Math.max(small?170:260,(fs.clientHeight-320)/rows):0;
  [].forEach.call($("gplot").querySelectorAll(".gp"),function(el,i){
   var p=panel(all[i],S.locs[i],ds,eff,small,Math.max(240,el.clientWidth),FH);p.miss.forEach(function(m){miss[m]=1});el.innerHTML=p.svg});
  var ms_=Object.keys(miss);
  if(ms_.length)$("gplot").insertAdjacentHTML("beforeend",'<p class=mut>No forecast for some nowcast dates: '+ms_.join(", ")+'</p>');
  wisChart(all[0],eff)}).catch(function(){$("gplot").innerHTML='<p class=mut>Could not load data.</p>'})}
function wisChart(d,eff){
 var W_=function(id){return ((d.score[id]||{})[TGT()]||{})[S.src]};
 var X_=function(id){return (((d.stats||{})[id]||{})[TGT()]||{})[S.src]};
 var sel=eff.map(function(e){return e.id}).filter(function(id){return W_(id)&&X_(id)});
 if(!sel.length){$("gwis").innerHTML='<p class=mut>No scores available.</p>';return}
 var com=Object.keys(W_(sel[0])).filter(function(o){return sel.every(function(id){return W_(id)[o]!==undefined&&X_(id)[o]!==undefined})});
 if(!com.length){$("gwis").innerHTML='<p class=mut>No nowcast dates common to the selected checkpoints.</p>';return}
 var mean=function(f){return sel.map(function(id){var a=com.map(function(o){return f(id,o)});return a.reduce(function(x,y){return x+y},0)/a.length})};
 // WIS parts by row, natural then log scale (log(x+1)), where every common date has them.
 var P_=function(id){return (((d.parts||{})[id]||{})[TGT()]||{})[S.src]};
 var parts=sel.map(function(id){var p=P_(id);if(!p||!com.every(function(o){return p[o]}))return null;
  return [0,1,2,3,4,5,6].map(function(k){return com.reduce(function(a,o){return a+p[o][k]},0)/com.length})});
 var PN=(D.parts||[]).map(function(p){return p[0]}),PS=(D.parts||[]).map(function(p){return p[1]});
 // A part's segment: white beneath, its row's colour at the part's share on top, so it shades alike in both themes.
 function seg(x,y,w,h,c,k,v,tot){if(w<=0)return "";var g=w>2?w-1:w;
  return '<rect x="'+x+'" y="'+y+'" width="'+g+'" height="'+h+'" fill="#fff" /><rect x="'+x+'" y="'+y+'" width="'+g+'" height="'+h+'" fill="'+c+'" fill-opacity="'+PS[k]+'" stroke="'+c+'" stroke-width=".6"><title>'+PN[k]+' '+fmt(v)+' of '+fmt(tot)+'</title></rect>'}
 function stack(id,y,vals,tot,mx,sc){var x=0,o='';vals.forEach(function(v,k){var w=v/mx*sc;o+=seg(x,y,w,12,colour(id),k,v,tot);x+=w});return o}
 function key(){return PN.map(function(n,k){return '<span class=lg><i style="display:inline-block;width:10px;height:10px;background:var(--fg);opacity:'+PS[k]+'"></i>'+n+'</span>'}).join("")}
 var wis=mean(function(id,o){return W_(id)[o]}),c50=mean(function(id,o){return X_(id)[o][0]}),c95=mean(function(id,o){return X_(id)[o][1]}),band=sel.map(function(id){return com.every(function(o){return X_(id)[o][3]==90})?90:95}),bias=mean(function(id,o){return X_(id)[o][2]});
 function shortLab(id){var m=C.meta.find(function(x){return x.id==id});if(!m)return id;var r=RUNS.find(function(r){return r.key==m.run});return r?r.label+" n"+m.n:m.label}
 // Each row's swatch is its plot line (colour and dash); the text stays in text colour.
 var rk={};eff.forEach(function(e){rk[e.id]=e.run=="ref"?0:e.rank});
 function lab(id,y){return '<line x1="0" x2="22" y1="'+(y+5)+'" y2="'+(y+5)+'" stroke="'+colour(id)+'" stroke-width="2.2"'+dashAttr(rk[id]||0)+' /><text x="28" y="'+(y+9)+'" class=ax>'+shortLab(id)+'</text>'}
 var W=Math.max(200,$("gwis").clientWidth),RH=30;
 function rows(title,body,H,k){return '<h3>'+title+ftInfo(k)+'</h3><svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'">'+body+'</svg>'}
 // mean WIS
 var mx=Math.max.apply(null,wis)*1.05,h1='';
 sel.forEach(function(id,i){var y=2+i*RH,w=wis[i]/mx*(W-48);
  h1+=lab(id,y)+(parts[i]?stack(id,y+12,parts[i].slice(0,3),wis[i],mx,W-48):'<rect x="0" y="'+(y+12)+'" width="'+w+'" height="12" fill="'+colour(id)+'" />')+'<text x="'+(w+4)+'" y="'+(y+22)+'" class=ax>'+fmt(wis[i])+'</text>'});
 // mean WIS on the log scale, where the rows have it
 var lg=sel.map(function(id,i){return parts[i]?parts[i][3]:null}),lmx=Math.max.apply(null,lg.filter(function(v){return v!==null}).concat([0]))*1.05,h4='';
 sel.forEach(function(id,i){if(lg[i]===null)return;var y=2+i*RH,w=lg[i]/lmx*(W-48);
  h4+=lab(id,y)+stack(id,y+12,parts[i].slice(4,7),lg[i],lmx,W-48)+'<text x="'+(w+4)+'" y="'+(y+22)+'" class=ax>'+fmt(lg[i])+'</text>'});
 // coverage of the 50% and outer intervals: 90% where the forecasts have
 // 5% and 95% quantiles, else 95% (the German hub's); nominal lines dashed
 var cw=W-36,h2='',outs=band.filter(function(b,i){return band.indexOf(b)==i}).sort();
 sel.forEach(function(id,i){var y=2+i*RH;
  h2+=lab(id,y)
   +'<rect x="0" y="'+(y+12)+'" width="'+c50[i]*cw+'" height="7" fill="'+colour(id)+'"><title>50% interval: '+c50[i].toFixed(2)+'</title></rect><text x="'+(c50[i]*cw+3)+'" y="'+(y+19)+'" class=ax>'+c50[i].toFixed(2)+'</text>'
   +'<rect x="0" y="'+(y+20)+'" width="'+c95[i]*cw+'" height="7" fill="'+colour(id)+'" opacity=".55"><title>'+band[i]+'% interval: '+c95[i].toFixed(2)+'</title></rect><text x="'+(c95[i]*cw+3)+'" y="'+(y+27)+'" class=ax>'+c95[i].toFixed(2)+'</text>'});
 var H2=4+sel.length*RH;
 h2+='<line x1="'+0.5*cw+'" x2="'+0.5*cw+'" y1="0" y2="'+H2+'" class=mark />';
 outs.forEach(function(b){h2+='<line x1="'+b/100*cw+'" x2="'+b/100*cw+'" y1="0" y2="'+H2+'" class=mark />'});
 // bias, -1 to 1 with 0 marked
 var bw=W,h3='';
 sel.forEach(function(id,i){var y=2+i*RH,x0=bw/2,w=Math.abs(bias[i])*(bw/2-34);
  h3+=lab(id,y)+'<rect x="'+(bias[i]>=0?x0:x0-w)+'" y="'+(y+12)+'" width="'+w+'" height="12" fill="'+colour(id)+'"><title>bias '+bias[i].toFixed(3)+' (positive: forecast above truth)</title></rect><text x="'+(bias[i]>=0?x0+w+3:x0-w-3)+'" y="'+(y+22)+'" text-anchor="'+(bias[i]>=0?"start":"end")+'" class=ax>'+bias[i].toFixed(2)+'</text>'});
 h3+='<line x1="'+bw/2+'" x2="'+bw/2+'" y1="0" y2="'+H2+'" stroke="var(--fg)" stroke-width="1" />';
 var any=parts.some(function(p){return p});
 $("gwis").innerHTML=(any?'<div style="font-size:11px">'+key()+'</div>':'')+rows("mean "+C.metric,h1,H2,"wis")+(any?rows("mean "+C.metric+" on the log scale, log(x + 1)",h4,H2,"logwis"):'')+rows("coverage 50% and "+outs.join("% or ")+"% (dashed: nominal)",h2,H2,"coverage")+rows("bias (+: forecast above truth)",h3,H2,"biasscore")
  +'<div class=mut style="font-size:11px">'+com.length+' common nowcast dates</div>';
}
function chips(){var h='';
 RUNS.forEach(function(r){if(!isOn(r.key))return;var all=ofRun(r.key),ch=chosen(r.key);
  h+='<span class=lg><b style="color:'+r.color+'">'+r.label+'</b> '+(all.length?all.map(function(m){var on=ch.indexOf(m.id)>=0;return '<button class="tog'+(on?' on':'')+'" data-chip="'+r.key+'|'+m.id+'" title="'+m.id+' ('+m.count+' dates)" style="'+(on?'background:'+r.color+';border-color:'+r.color:'')+'">n'+m.n+'</button>'}).join(""):'<span class=mut>no checkpoints scored yet</span>')+'</span>'});
 $("gck").innerHTML=h}
function locBtns(){
 $("glocs").innerHTML=C.groups.filter(function(g){return g.locs.length}).map(function(g){return '<div><span class=mut>'+g.name+'</span> '+g.locs.map(function(l){return '<button class="tog'+(S.locs.indexOf(l)>=0?' on':'')+'" data-loc="'+l+'">'+l+'</button>'}).join("")+'</div>'}).join("");
 $("gsum").textContent="Locations: "+S.locs.join(", ")}
function draw(){
 if(C.empty)return empty();
 $("gdate").textContent=ORI()[S.oi]+(S.latest?" (latest)":"");
 $("gsl").value=S.oi;$("gnd").value=S.nd;$("gstep").value=S.step;$("gsrc").value=S.src;$("gyfit").value=S.yfit;$("gylog").checked=S.ylog;$("gbands").value=S.bands;
 $("gtgtl").hidden=(C.targets||[]).length<2;
 $("gtgt").innerHTML=(C.targets||[]).map(function(v){return '<option value="'+v+'"'+(v==TGT()?' selected':'')+'>'+tlab(v)+'</option>'}).join("");
 chips();locBtns();plots();save()}
function build(){
 if(DSS.length>1){$("gdsl").hidden=false;$("gds").innerHTML=DSS.map(function(d){return '<option value="'+d.key+'"'+(d.key==S.ds?' selected':'')+'>'+d.label+(d.empty?' (no nowcasts yet)':'')+'</option>'}).join("");
  $("gds").onchange=function(){S.ds=this.value;C=cur();S.locs=C.locs.length?[C.locs[0]]:[];S.oi=ORI().length-1;S.latest=true;$("gsl").max=Math.max(0,S.oi);draw()}}
 $("gsl").min=0;$("gsl").max=ORI().length-1;
 $("gsrc").onchange=function(){S.src=this.value;draw()};
 $("gtgt").onchange=function(){S.target=this.value;draw()};
 $("gnd").onchange=function(){S.nd=+this.value;draw()};
 $("gstep").onchange=function(){S.step=+this.value;draw()};
 $("gyfit").onchange=function(){S.yfit=+this.value;draw()};
 $("gylog").onchange=function(){S.ylog=this.checked;draw()};
 $("gbands").onchange=function(){S.bands=+this.value;draw()};
 var pl=timePlayer({id:"german",btn:$("gplay"),speed:$("gspd"),len:function(){return ORI().length},
  get:function(){return S.oi},set:function(i){S.oi=i;S.latest=i==ORI().length-1;draw()}});
 $("gsl").oninput=function(){pl.pause();S.oi=+this.value;S.latest=S.oi==ORI().length-1;draw()};
 $("gprev").onclick=function(){S.oi=Math.max(0,S.oi-1);S.latest=false;draw()};
 $("gnext").onclick=function(){S.oi=Math.min(ORI().length-1,S.oi+1);S.latest=S.oi==ORI().length-1;draw()};
 $("glocs").onclick=function(e){var b=e.target.closest("button");if(!b)return;var l=b.dataset.loc,i=S.locs.indexOf(l);
  if(i>=0){if(S.locs.length>1)S.locs.splice(i,1)}else S.locs.push(l);draw()};
 $("gck").onclick=function(e){var b=e.target.closest("button");if(!b)return;var p=b.dataset.chip.split("|"),k=p[0],id=p[1];
  var ids=chosen(k).slice(),i=ids.indexOf(id);if(i>=0){if(ids.length>1)ids.splice(i,1)}else ids.push(id);S.chk[k]=ids;draw()};
 draw()}
window.addEventListener("ftsel",function(){draw()});
// New data from the server (live.js): keep every control, drop cached locations.
window.addEventListener("ftdata",function(e){if(e.detail!=="nowcasts")return;D=JSON.parse($("gdata").textContent);DSS=D.datasets;RUNS=D.runs;C=cur();cache={};if(!window.ftPlaying)draw()});
var rsz=null;window.addEventListener("resize",function(){clearTimeout(rsz);rsz=setTimeout(plots,150)});
var sec=document.getElementById("nowcasts");if(sec)sec.addEventListener("toggle",function(){if(sec.open)plots()});
build();
})();
