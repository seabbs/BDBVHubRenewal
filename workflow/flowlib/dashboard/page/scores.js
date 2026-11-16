(function(){
var P=JSON.parse(document.getElementById("sdata").textContent);
var $=function(i){return document.getElementById(i)};
var SERS=P.series.map(function(s){return [s.key,s.label]}),RUNK=P.series.filter(function(s){return s.run}).map(function(s){return s.key});
var LV={germany:[["national","national"],["states","states"],["age","ages"]],ebola:[["national","national"],["provinces","provinces"]]};
var MK=["circle","square","triangle","diamond"];
var S={study:"germany",src:"daily",lv:["national"],cmp:false,log:false,ser:P.defaults.slice()};
try{var sv=JSON.parse(sessionStorage.getItem("fts3")||"null");if(sv)S=sv}catch(e){}
function save(){try{sessionStorage.setItem("fts3",JSON.stringify(S))}catch(e){}}
function mark(k,x,y,c,t){var r=3.6,s="";
 if(k=="circle")s='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+c+'">';
 else if(k=="square")s='<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+c+'">';
 else if(k=="triangle")s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+(y+r)+' '+(x-r-1)+','+(y+r)+'" fill="'+c+'">';
 else s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+y+' '+x+','+(y+r+1)+' '+(x-r-1)+','+y+'" fill="'+c+'">';
 return s+'<title>'+t+'</title>'+(k=="circle"?"</circle>":k=="square"?"</rect>":"</polygon>")}
function btn(id,label,on,sty){return '<button data-id="'+id+'" class="tog'+(on?" on":"")+'"'+(sty?' style="'+sty+'"':"")+'>'+label+'</button>'}
function controls(){
 var lv=LV[S.study];
 $("sctl").innerHTML=[["germany","Germany"],["ebola","BDBV"]].map(function(a){return btn("study:"+a[0],a[1],S.study==a[0])}).join("")
 +'<span class=sep></span>'+["daily","weekly"].map(function(a){return btn("src:"+a,a,S.src==a)}).join("")
 +'<span class=sep></span>'+lv.map(function(a){return btn("lv:"+a[0],a[1],S.lv.indexOf(a[0])>=0)}).join("")+btn("cmp","compare levels",S.cmp)
 +'<span class=sep></span>'+SERS.map(function(a){return btn("ser:"+a[0],a[1],S.ser.indexOf(a[0])>=0,"border-left:4px solid "+P.col[a[0]])}).join("")
 +'<span class=sep></span>'+btn("log","log scale",S.log)}
function tick(arr,v){var i=arr.indexOf(v);if(i>=0)arr.splice(i,1);else arr.push(v)}
$("sctl").onclick=function(e){var b=e.target.closest("button");if(!b)return;var p=b.dataset.id.split(":"),k=p[0],v=p[1];
 if(k=="study"){S.study=v;S.lv=S.lv.filter(function(l){return LV[v].some(function(a){return a[0]==l})});if(!S.lv.length)S.lv=["national"]}
 else if(k=="src")S.src=v;
 else if(k=="lv"){if(S.cmp){tick(S.lv,v);if(!S.lv.length)S.lv=[v]}else S.lv=[v]}
 else if(k=="cmp"){S.cmp=!S.cmp;if(!S.cmp)S.lv=[S.lv[0]]}
 else if(k=="log")S.log=!S.log;
 else tick(S.ser,v);
 save();draw()};
function fmt(v){return (+v).toPrecision(3)}
function niceTicks(lo,hi){var st=(hi-lo)/4,m=Math.pow(10,Math.floor(Math.log10(st))),k=[1,2,2.5,5,10].find(function(x){return x*m>=st})*m,o=[];
 for(var v=Math.ceil(lo/k)*k;v<=hi+1e-9;v+=k)o.push(+v.toFixed(6));return o}
function draw(){
 controls();
 var lv=LV[S.study].filter(function(a){return S.lv.indexOf(a[0])>=0});
 var rows=P.rows.filter(function(r){return r[0]==S.study&&r[2]==S.src});
 var shown=rows.filter(function(r){return S.ser.indexOf(r[3])>=0&&r[3]!="rw"&&lv.some(function(l){return l[0]==r[1]})});
 var W=640,H=320,L=44,R=16,T=10,B=44;
 var ns=shown.filter(function(r){return r[4]>0}).map(function(r){return r[4]});
 var nmin=ns.length?Math.min.apply(null,ns):250,nmax=ns.length?Math.max.apply(null,ns):250;
 var zx=nmin/2,x0=Math.log2(zx),x1=Math.log2(nmax*1.18);
 var sx=function(n){return L+(Math.log2(n||zx)-x0)/(x1-x0)*(W-L-R)};
 var vals=shown.map(function(r){return r[6]});
 var d=Math.max.apply(null,[0.05].concat(vals.map(function(v){return Math.abs(v-1)})))*1.1,lo,hi,sy;
 if(S.log){var q=Math.max.apply(null,[1.05].concat(vals.map(function(v){return Math.max(v,1/v)})))*1.1;lo=Math.log(1/q);hi=Math.log(q);sy=function(v){return T+(1-(Math.log(v)-lo)/(hi-lo))*(H-T-B)}}
 else{lo=Math.max(0,1-d);hi=1+d;sy=function(v){return T+(1-(v-lo)/(hi-lo))*(H-T-B)}}
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var yt=S.log?[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.25,1.5,2,3,4,6,8].filter(function(v){return Math.log(v)>=lo&&Math.log(v)<=hi}):niceTicks(lo,hi);
 yt.forEach(function(v){h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-5)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+v+'</text>'});
 var xt=[[zx,"zero-shot"]].concat(Array.from(new Set(ns)).sort(function(a,b){return a-b}).map(function(n){return [n,""+n]}));
 xt.forEach(function(t){h+='<line x1="'+sx(t[0])+'" x2="'+sx(t[0])+'" y1="'+T+'" y2="'+(H-B)+'" class=grid /><text x="'+sx(t[0])+'" y="'+(H-B+14)+'" text-anchor=middle class=ax>'+t[1]+'</text>'});
 h+='<text x="'+(W/2)+'" y="'+(H-6)+'" text-anchor=middle class=ax>simulations (log2)</text>';
 P.pending.forEach(function(p){if(p.n<=nmax*1.18&&p.n>=zx){h+='<line x1="'+sx(p.n)+'" x2="'+sx(p.n)+'" y1="'+T+'" y2="'+(H-B)+'" class=mark /><text x="'+(sx(p.n)+3)+'" y="'+(T+10)+'" class=ax>n'+p.n+' scoring</text>'}});
 if(S.ser.indexOf("rw")>=0)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(1)+'" y2="'+sy(1)+'" stroke="'+P.col.rw+'" stroke-width=2 ><title>randomwalknowcast = 1</title></line>';
 var any=false,sum=[];
 lv.forEach(function(l,li){var mk=MK[li];
  RUNK.concat(["zero","complete"]).forEach(function(s){
   if(S.ser.indexOf(s)<0)return;
   var rs=rows.filter(function(r){return r[1]==l[0]&&r[3]==s}).sort(function(a,b){return a[4]-b[4]});
   if(!rs.length)return;any=true;var c=P.col[s];
   var lab=SERS.find(function(x){return x[0]==s})[1];
   if(s=="zero"||s=="complete"){var v=rs[0][6];
    h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" stroke="'+c+'" stroke-width=1.5 stroke-dasharray="'+(s=="zero"?"6 3":"2 3")+'" />';
    h+=mark(mk,sx(zx),sy(v),c,l[1]+", "+lab+": "+fmt(v))}
   else{
    var zs=rows.filter(function(r){return r[1]==l[0]&&r[3]=="zero"})[0];
    var pts=(zs?[[zx,zs[6]]]:[]).concat(rs.map(function(r){return [r[4],r[6]]}));
    h+='<polyline points="'+pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="'+c+'" stroke-width=1.8 />';
    rs.forEach(function(r){h+=mark(mk,sx(r[4]),sy(r[6]),c,l[1]+", "+lab+" n"+r[4]+" "+r[5]+": "+fmt(r[6]));
     if(s==RUNK[0]&&lv.length==1)h+='<text x="'+(sx(r[4])+6)+'" y="'+(sy(r[6])-6)+'" class=ax>'+fmt(r[6])+'</text>'});
    if(RUNK.indexOf(s)>=0){var last=rs[rs.length-1];
     sum.push((lv.length>1?l[1]+": ":"")+lab+" n"+last[4]+": "+fmt(last[6])+(zs?" vs zero-shot "+fmt(zs[6])+", "+((last[6]/zs[6]-1)*100>=0?"+":"")+((last[6]/zs[6]-1)*100).toFixed(0)+"%":""))}}})});
 h+='</svg>';
 var leg=SERS.filter(function(s){return S.ser.indexOf(s[0])>=0}).map(function(s){return '<span><i style="background:'+P.col[s[0]]+'"></i>'+s[1]+(s[0]=="zero"?" (dashed)":s[0]=="complete"?" (dotted)":"")+'</span>'}).join("");
 if(lv.length>1)leg+=lv.map(function(l,i){return '<span><svg width=12 height=12 viewBox="0 0 12 12">'+mark(MK[i],6,6,"var(--fg)","")+'</svg>'+l[1]+'</span>'}).join("");
 var note="";
 if(S.ser.indexOf("complete")>=0&&!rows.some(function(r){return r[3]=="complete"}))note+="<p class=mut>complete-only pending</p>";
 if(P.pending.length)note+='<p class=mut>Saved, not yet in the scores: '+P.pending.map(function(p){return p.run+" n"+p.n+(p.hash?" "+p.hash:"")}).join(", ")+'.</p>';
 if(!any)note+="<p class=mut>No rows for this selection.</p>";
 $("splot").innerHTML='<div class=sum>'+(sum.join("; ")||"")+'</div>'+h+'<div class=leg>'+leg+'</div>'+note+'<p class=mut>Relative to randomwalknowcast ('+(S.study=="germany"?"WIS":"CRPS")+', below 1 is better). Marker shape shows level.</p>';
}
draw();
})();
