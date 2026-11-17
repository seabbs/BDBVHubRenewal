(function(){
var D=JSON.parse(document.getElementById("gdata").textContent);
var DSS=D.datasets,RUNS=D.runs,ST=D.style;
// Quantile index of a level, and the bands, as the paper nowcast panels.
function QI(l){return ST.levels.indexOf(l)}
var $=function(i){return document.getElementById(i)};
var DASH=["","6 3","8 3 2 3","2 3"],MK=["circle","square","triangle","diamond"];
var S={ds:DSS[0].key,locs:["DE"],src:"daily",oi:null,latest:true,nd:1,step:2,on:null,chk:{}};
var KEY="ftg4";
try{var sv=JSON.parse(sessionStorage.getItem(KEY)||"null");if(sv)Object.keys(sv).forEach(function(k){S[k]=sv[k]})}catch(e){}
function cur(){return DSS.find(function(d){return d.key==S.ds})||DSS[0]}
var C=cur();S.ds=C.key;
function ofRun(k){return C.meta.filter(function(m){return m.run==k}).sort(function(a,b){return a.n-b.n})}
function refs(){return C.meta.filter(function(m){return m.run=="ref"})}
function ORI(){return C.origins}
S.locs=(S.locs||[]).filter(function(l){return C.locs.indexOf(l)>=0});if(!S.locs.length)S.locs=[C.locs[0]];
if(S.latest||ORI().indexOf(S.origin)<0){S.oi=ORI().length-1;S.latest=true}else S.oi=ORI().indexOf(S.origin);
if(!S.on){var r0=RUNS.find(function(r){return ofRun(r.key).length})||RUNS[0];
 S.on=[r0?r0.key:"","zero-shot"].filter(function(x){return x})}
function save(){try{sessionStorage.setItem(KEY,JSON.stringify({ds:S.ds,locs:S.locs,src:S.src,latest:S.latest,origin:ORI()[S.oi],nd:S.nd,step:S.step,on:S.on,chk:S.chk}))}catch(e){}}
function chosen(k){var all=ofRun(k),ids=(S.chk[k]||[]).filter(function(i){return all.some(function(m){return m.id==i})});
 if(!ids.length&&all.length)ids=[all[all.length-1].id];return ids}
function effective(){var o=[];
 RUNS.forEach(function(r){if(S.on.indexOf(r.key)>=0){var ids=chosen(r.key);
  ids.forEach(function(id,i){o.push({id:id,run:r.key,rank:ids.length-1-i})})}});
 refs().forEach(function(m){if(S.on.indexOf(m.id)>=0)o.push({id:m.id,run:"ref",rank:0})});
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
function fmt(v){return v>=1000?Math.round(v).toLocaleString("en-GB"):(+v).toPrecision(3)}
function dashAttr(rank){var d=DASH[rank%4];return d?' stroke-dasharray="'+d+'"':""}
function mark(k,x,y,c,t,r){r=r||3;var s="";
 if(k=="circle")s='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+c+'">';
 else if(k=="square")s='<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+c+'">';
 else if(k=="triangle")s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+(y+r)+' '+(x-r-1)+','+(y+r)+'" fill="'+c+'">';
 else s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+y+' '+x+','+(y+r+1)+' '+(x-r-1)+','+y+'" fill="'+c+'">';
 return s+'<title>'+t+'</title>'+(k=="circle"?"</circle>":k=="square"?"</rect>":"</polygon>")}
function panel(d,loc,ds,eff,small){
 var W=small?300:640,H=small?200:320,L=small?36:48,R=8,T=small?8:10,B=small?24:34,ymax=1,i;
 var t0=ms(ds[0])-70*864e5,t1=ms(ds[ds.length-1])+35*864e5;
 var tr=d.truth,pts=[];
 for(i=0;i<tr.t.length;i++){var x=ms(tr.t[i]);if(x>=t0&&x<=t1){pts.push([x,tr.y[i]]);ymax=Math.max(ymax,tr.y[i])}}
 var fs=[];
 eff.forEach(function(e){ds.forEach(function(o,di){var s=(d.series[e.id]||{})[S.src],f=s&&s[o];
  if(f){fs.push({e:e,f:f,di:di,o:o});f.q.forEach(function(q){ymax=Math.max(ymax,q[q.length-1])})}})});
 var runBands=eff.some(function(e){return e.run!="ref"&&fs.some(function(z){return z.e===e})});
 ymax*=1.06;
 var sx=function(x){return L+(x-t0)/(t1-t0)*(W-L-R)},sy=function(y){return T+(1-y/ymax)*(H-T-B)};
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var step=Math.pow(10,Math.floor(Math.log10(ymax/4)));
 var m=[1,2,2.5,5,10].find(function(k){return k*step>=ymax/(small?3:5)})*step;
 for(var v=0;v<=ymax;v+=m)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-4)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+fmt(v)+'</text>';
 var xs=small?28:14;
 for(var x=t0;x<=t1;x+=xs*864e5)h+='<text x="'+sx(x)+'" y="'+(H-B+12)+'" text-anchor=middle class=ax>'+new Date(x).toISOString().slice(5,10)+'</text>';
 ds.forEach(function(o){h+='<line x1="'+sx(ms(o))+'" x2="'+sx(ms(o))+'" y1="'+T+'" y2="'+(H-B)+'" class=mark />'});
 fs.forEach(function(z){var e=z.e,f=z.f,c=colour(e.id);
  var P=function(k){return f.t.map(function(d,j){return sx(ms(d))+","+sy(f.q[j][k])})};
  var latest=eff.filter(function(q){return q.run==e.run}).every(function(q){return q.rank>=e.rank});
  if(e.rank==0&&(e.run!="ref"||!runBands)){var op=ds.length>1?.5:1;
   ST.bands.forEach(function(b){h+='<polygon points="'+P(QI(b[0])).concat(P(QI(b[1])).reverse()).join(" ")+'" fill="'+c+'" opacity="'+b[2]*op+'" />'})}
  h+='<polyline points="'+P(QI(0.5)).join(" ")+'" fill=none stroke="'+c+'" stroke-width='+(small?1.5:2)+dashAttr(e.run=="ref"?0:e.rank)+' />';
  f.t.forEach(function(d,j){h+=mark(MK[z.di%4],sx(ms(d)),sy(f.q[j][QI(0.5)]),c,id2lab(e.id)+' (nowcast '+z.o+') '+d+' median '+fmt(f.q[j][QI(0.5)])+' (50%: '+fmt(f.q[j][QI(0.25)])+' to '+fmt(f.q[j][QI(0.75)])+'; 95%: '+fmt(f.q[j][QI(0.025)])+' to '+fmt(f.q[j][QI(0.975)])+')',small?2.2:2.8)})});
 if(pts.length){h+='<polyline points="'+pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="var(--truth)" stroke-width=1.5 />';
  pts.forEach(function(p){h+='<circle cx="'+sx(p[0])+'" cy="'+sy(p[1])+'" r=1.8 fill="var(--truth)"><title>truth '+new Date(p[0]).toISOString().slice(0,10)+': '+fmt(p[1])+'</title></circle>'})}
 return {svg:h+'</svg>',miss:eff.filter(function(e){return !fs.some(function(z){return z.e===e})}).map(function(e){return id2lab(e.id)})};
}
function legend(eff,ds){
 function sw(c,rank){return '<svg width=26 height=8 viewBox="0 0 26 8"><line x1=0 x2=26 y1=4 y2=4 stroke="'+c+'" stroke-width=2'+dashAttr(rank)+' /></svg>'}
 var leg='<div class=lg><b>truth</b> '+sw("var(--fg)",0)+'</div>';
 RUNS.forEach(function(r){var g=eff.filter(function(e){return e.run==r.key});if(!g.length)return;
  leg+='<div class=lg><b style="color:'+r.color+'">'+r.label+'</b> '+g.map(function(e){return sw(colour(e.id),e.rank)+"n"+C.meta.find(function(x){return x.id==e.id}).n}).join(" ")+'</div>'});
 var rf=eff.filter(function(e){return e.run=="ref"});
 if(rf.length)leg+='<div class=lg><b>reference</b> '+rf.map(function(e){return sw(colour(e.id),0)+id2lab(e.id)}).join(" ")+'</div>';
 if(ds.length>1)leg+='<div class=lg><b>nowcast date</b> '+ds.map(function(o,i){return '<svg width=12 height=12 viewBox="0 0 12 12">'+mark(MK[i%4],6,6,"var(--fg)","",3)+'</svg>'+o}).join(" ")+'</div>';
 return '<div class=leg2>'+leg+'</div>'}
function plots(){
 var eff=effective(),ds=dates(),small=S.locs.length>1;
 Promise.all(S.locs.map(load)).then(function(all){
  var miss={},h='<div class=panels style="grid-template-columns:repeat('+(small?"auto-fit,minmax(260px,1fr)":"1,1fr")+')">';
  all.forEach(function(d,i){var p=panel(d,S.locs[i],ds,eff,small);p.miss.forEach(function(m){miss[m]=1});
   h+='<div>'+(small?'<h4>'+S.locs[i]+'</h4>':"")+p.svg+'</div>'});
  h+='</div>'+legend(eff,ds);
  var ms_=Object.keys(miss);
  if(ms_.length)h+='<p class=mut>No forecast for some nowcast dates: '+ms_.join(", ")+'</p>';
  h+='<p class=mut>Weekly counts by target week. Colour is the run. Line style runs from dotted (earliest) to solid (latest selected checkpoint). Bands (50% dark, 95% light) are drawn for the latest checkpoint of each run. Truth: '+C.truth+'.'+(ds.length>1?' Marker shape shows the nowcast date, and each nowcast ends at its date.':'')+'</p>';
  $("gplot").innerHTML=h;
  wisChart(all[0],eff)}).catch(function(){$("gplot").innerHTML='<p class=mut>Could not load data.</p>'})}
function wisChart(d,eff){
 var sel=eff.map(function(e){return e.id}).filter(function(id){return d.wis[id]&&d.wis[id][S.src]});
 if(!sel.length){$("gwis").innerHTML='<p class=mut>No WIS available.</p>';return}
 var com=Object.keys(d.wis[sel[0]][S.src]).filter(function(o){return sel.every(function(id){return d.wis[id][S.src][o]!==undefined})});
 if(!com.length){$("gwis").innerHTML='<p class=mut>No nowcast dates common to the selected checkpoints.</p>';return}
 var vals=sel.map(function(id){var a=com.map(function(o){return d.wis[id][S.src][o]});return a.reduce(function(x,y){return x+y},0)/a.length});
 var W=520,H=60+sel.length*26,L=230,mx=Math.max.apply(null,vals)*1.05,h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 sel.forEach(function(id,i){var y=10+i*26,w=vals[i]/mx*(W-L-60);
  h+='<text x="'+(L-6)+'" y="'+(y+14)+'" text-anchor=end class=ax>'+id2lab(id)+'</text><rect x="'+L+'" y="'+y+'" width="'+w+'" height="18" fill="'+colour(id)+'" /><text x="'+(L+w+4)+'" y="'+(y+14)+'" class=ax>'+fmt(vals[i])+'</text>'});
 $("gwis").innerHTML=h+'</svg><p class=mut>Mean WIS for '+S.locs[0]+' ('+S.src+' source) over '+com.length+' nowcast dates common to the selected checkpoints. Lower is better.</p>';
}
function chips(){var h='';
 RUNS.forEach(function(r){if(S.on.indexOf(r.key)<0)return;var all=ofRun(r.key),ch=chosen(r.key);
  h+='<span class=lg><b style="color:'+r.color+'">'+r.label+'</b> '+(all.length?all.map(function(m){var on=ch.indexOf(m.id)>=0;return '<button class="tog'+(on?' on':'')+'" data-chip="'+r.key+'|'+m.id+'" title="'+m.id+' ('+m.count+' dates)" style="'+(on?'background:'+r.color+';border-color:'+r.color:'')+'">n'+m.n+'</button>'}).join(""):'<span class=mut>no checkpoints scored yet</span>')+'</span>'});
 $("gck").innerHTML=h}
function runBtns(){
 $("gruns").innerHTML=RUNS.map(function(r){return '<button class="tog'+(S.on.indexOf(r.key)>=0?' on':'')+'" data-run="'+r.key+'" style="border-left:4px solid '+r.color+'">'+r.label+'</button>'}).join("")
  +refs().map(function(m){return '<button class="tog'+(S.on.indexOf(m.id)>=0?' on':'')+'" data-run="'+m.id+'" style="border-left:4px solid '+m.color+'">'+m.label+'</button>'}).join("")}
function locBtns(){
 $("glocs").innerHTML=C.groups.filter(function(g){return g.locs.length}).map(function(g){return '<div><span class=mut>'+g.name+'</span> '+g.locs.map(function(l){return '<button class="tog'+(S.locs.indexOf(l)>=0?' on':'')+'" data-loc="'+l+'">'+l+'</button>'}).join("")+'</div>'}).join("");
 $("gsum").textContent="Locations: "+S.locs.join(", ")}
function draw(){
 $("gdate").textContent=ORI()[S.oi]+(S.latest?" (latest)":"");
 $("gsl").value=S.oi;$("gnd").value=S.nd;$("gstep").value=S.step;$("gsrc").value=S.src;
 runBtns();chips();locBtns();plots();save()}
function build(){
 if(DSS.length>1){$("gdsl").hidden=false;$("gds").innerHTML=DSS.map(function(d){return '<option value="'+d.key+'"'+(d.key==S.ds?' selected':'')+'>'+d.label+'</option>'}).join("");
  $("gds").onchange=function(){S.ds=this.value;C=cur();S.locs=[C.locs[0]];S.oi=ORI().length-1;S.latest=true;$("gsl").max=S.oi;draw()}}
 $("gsl").min=0;$("gsl").max=ORI().length-1;
 $("gsrc").onchange=function(){S.src=this.value;draw()};
 $("gnd").onchange=function(){S.nd=+this.value;draw()};
 $("gstep").onchange=function(){S.step=+this.value;draw()};
 var pl=timePlayer({id:"german",btn:$("gplay"),speed:$("gspd"),len:function(){return ORI().length},
  get:function(){return S.oi},set:function(i){S.oi=i;S.latest=i==ORI().length-1;draw()}});
 $("gsl").oninput=function(){pl.pause();S.oi=+this.value;S.latest=S.oi==ORI().length-1;draw()};
 $("gprev").onclick=function(){S.oi=Math.max(0,S.oi-1);S.latest=false;draw()};
 $("gnext").onclick=function(){S.oi=Math.min(ORI().length-1,S.oi+1);S.latest=S.oi==ORI().length-1;draw()};
 $("glocs").onclick=function(e){var b=e.target.closest("button");if(!b)return;var l=b.dataset.loc,i=S.locs.indexOf(l);
  if(i>=0){if(S.locs.length>1)S.locs.splice(i,1)}else S.locs.push(l);draw()};
 $("gruns").onclick=function(e){var b=e.target.closest("button");if(!b)return;var k=b.dataset.run,i=S.on.indexOf(k);
  if(i>=0)S.on.splice(i,1);else{S.on.push(k);delete S.chk[k]}draw()};
 $("gck").onclick=function(e){var b=e.target.closest("button");if(!b)return;var p=b.dataset.chip.split("|"),k=p[0],id=p[1];
  var ids=chosen(k).slice(),i=ids.indexOf(id);if(i>=0){if(ids.length>1)ids.splice(i,1)}else ids.push(id);S.chk[k]=ids;draw()};
 draw()}
build();
})();
