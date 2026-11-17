(function(){
var P=JSON.parse(document.getElementById("sdata").textContent);
// A reference's role: baseline (scores are relative to it), start (where
// each run's curve starts, dashed) or other (dotted); defaults from config.
var ROLE=P.roles||{},DEF=P.view_defaults||{};
var $=function(i){return document.getElementById(i)};
var SERS=P.series.map(function(s){return [s.key,s.label]}),RUNK=P.series.filter(function(s){return s.run}).map(function(s){return s.key});
var STUDIES=P.studies.map(function(s){return [s,P.studylab[s]||s]});
function LVL(st){return (P.levels[st]||[]).map(function(l){return [l,l=="age"?"ages":l]})}
var LV={};P.studies.forEach(function(s){LV[s]=LVL(s)});
var MK=["circle","square","triangle","diamond"];
var VDASH=["","6 3","2 3","8 3 2 3"],REFK=Object.keys(ROLE);
var S={study:STUDIES.length?STUDIES[0][0]:"",vars:null,target:null,metric:null,lv:[DEF.level||""],cmp:false,log:false};
function SEL(){return ftSel.get()}
try{var sv=JSON.parse(sessionStorage.getItem("fts5")||"null");if(sv)S=sv}catch(e){}
function save(){try{sessionStorage.setItem("fts5",JSON.stringify(S))}catch(e){}}
function mark(k,x,y,c,t){var r=3.6,s="";
 if(k=="circle")s='<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+c+'">';
 else if(k=="square")s='<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+c+'">';
 else if(k=="triangle")s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+(y+r)+' '+(x-r-1)+','+(y+r)+'" fill="'+c+'">';
 else s='<polygon points="'+x+','+(y-r-1)+' '+(x+r+1)+','+y+' '+x+','+(y+r+1)+' '+(x-r-1)+','+y+'" fill="'+c+'">';
 return s+'<title>'+t+'</title>'+(k=="circle"?"</circle>":k=="square"?"</rect>":"</polygon>")}
function btn(id,label,on,sty){return '<button data-id="'+id+'" class="tog'+(on?" on":"")+'"'+(sty?' style="'+sty+'"':"")+'>'+label+'</button>'}
function MTS(){return P.metrics[S.study]||[]}
function MT(){var a=MTS();return a.indexOf(S.metric)>=0?S.metric:(a.indexOf(DEF.metric)>=0?DEF.metric:a[0]||"")}
function TGTS(){return P.targets[S.study]||[]}
function TGT(){var a=TGTS();return a.indexOf(S.target)>=0?S.target:(a.indexOf(DEF.target)>=0?DEF.target:a[0]||"")}
function tlab(v){return v?v.replace(/^(\d+)day$/,"$1-day"):"all"}
function VARS(){return P.variants[S.study]||[]}
function baseOf(v){return P.vbase[v]||v}
function curVars(){var av=VARS(),sel=(S.vars||[]).filter(function(v){return av.indexOf(v)>=0});
 if(!sel.length)sel=av.indexOf(DEF.source)>=0?[DEF.source]:av.slice(0,1);
 return sel}
function controls(){
 var lv=LV[S.study];
 $("sctl").innerHTML=STUDIES.map(function(a){return btn("study:"+a[0],a[1],S.study==a[0])}).join("")
 +(MTS().length>1?'<span class=sep></span><span class=mut>score</span>'+MTS().map(function(a){return btn("met:"+a,a.toUpperCase(),MT()==a)}).join(""):"")
 +(TGTS().length>1?'<span class=sep></span><span class=mut>target</span>'+TGTS().map(function(a){return btn("tgt:"+a,tlab(a),TGT()==a)}).join(""):"")
 +'<span class=sep></span><span class=mut>'+(P.vlabel[S.study]||"source")+'</span>'+VARS().map(function(a){return btn("var:"+a,a,curVars().indexOf(a)>=0)}).join("")
 +'<span class=sep></span>'+lv.map(function(a){return btn("lv:"+a[0],a[1],S.lv.indexOf(a[0])>=0)}).join("")+btn("cmp","compare levels",S.cmp)

 +'<span class=sep></span>'+btn("log","log scale",S.log)}
function tick(arr,v){var i=arr.indexOf(v);if(i>=0)arr.splice(i,1);else arr.push(v)}
$("sctl").onclick=function(e){var b=e.target.closest("button");if(!b)return;var p=b.dataset.id.split(":"),k=p[0],v=p[1];
 if(k=="study"){S.study=v;S.vars=null;S.lv=S.lv.filter(function(l){return LV[v].some(function(a){return a[0]==l})});if(!S.lv.length)S.lv=[LV[v].length?LV[v][0][0]:(DEF.level||"")]}
 else if(k=="tgt")S.target=v;
 else if(k=="met")S.metric=v;
 else if(k=="var"){var cv=curVars().slice();var i=cv.indexOf(v);if(i>=0){if(cv.length>1)cv.splice(i,1)}else cv.push(v);S.vars=cv}
 else if(k=="lv"){if(S.cmp){tick(S.lv,v);if(!S.lv.length)S.lv=[v]}else S.lv=[v]}
 else if(k=="cmp"){S.cmp=!S.cmp;if(!S.cmp)S.lv=[S.lv[0]]}
 else if(k=="log")S.log=!S.log;
 else{}
 save();draw()};
function fmt(v){return (+v).toPrecision(3)}
function niceTicks(lo,hi){var st=(hi-lo)/4,m=Math.pow(10,Math.floor(Math.log10(st))),k=[1,2,2.5,5,10].find(function(x){return x*m>=st})*m,o=[];
 for(var v=Math.ceil(lo/k)*k;v<=hi+1e-9;v+=k)o.push(+v.toFixed(6));return o}
function draw(){
 var avl=LV[S.study]||[];
 if(avl.length&&!avl.some(function(a){return S.lv.indexOf(a[0])>=0}))S.lv=[avl[0][0]];
 controls();
 var lv=avl.filter(function(a){return S.lv.indexOf(a[0])>=0});
 var vs=curVars();
 var TG=TGT(),MM=MT();
 function rowsFor(v){return P.rows.filter(function(r){return r[0]==S.study&&(r[7]==TG||r[7]=="")&&r[8]==MM&&(r[2]==v||(REFK.indexOf(r[3])>=0&&r[2]==baseOf(v)))})}
 var rows=[];vs.forEach(function(v){rowsFor(v).forEach(function(r){if(rows.indexOf(r)<0)rows.push(r)})});
 var shown=rows.filter(function(r){return SEL().indexOf(r[3])>=0&&ROLE[r[3]]!="baseline"&&lv.some(function(l){return l[0]==r[1]})});
 var W=640,H=320,L=44,R=16,T=10,B=44;
 var ns=shown.filter(function(r){return r[4]>0}).map(function(r){return r[4]});
 var nmin=ns.length?Math.min.apply(null,ns):250,nmax=ns.length?Math.max.apply(null,ns):250;
 var zx=nmin/2,x0=Math.log2(zx),x1=Math.log2(nmax*1.18);
 var sx=function(n){return L+(Math.log2(n||zx)-x0)/(x1-x0)*(W-L-R)};
 var vals=shown.map(function(r){return r[6]});
 // y range from the data shown; the randomwalknowcast line (1) counts only if close
 if(!vals.length)vals=[1];
 var vmin=Math.min.apply(null,vals),vmax=Math.max.apply(null,vals),spn=(vmax-vmin)||0.1*vmax;
 if(1>=vmin-spn&&1<=vmax+spn){vmin=Math.min(vmin,1);vmax=Math.max(vmax,1)}
 var showRw=1>=vmin&&1<=vmax,lo,hi,sy;
 if(S.log){lo=Math.log(vmin)-0.1*(Math.log(vmax)-Math.log(vmin)||0.1);hi=Math.log(vmax)+0.1*(Math.log(vmax)-Math.log(vmin)||0.1);sy=function(v){return T+(1-(Math.log(v)-lo)/(hi-lo))*(H-T-B)}}
 else{var pd=0.1*((vmax-vmin)||0.1*vmax);lo=Math.max(0,vmin-pd);hi=vmax+pd;sy=function(v){return T+(1-(v-lo)/(hi-lo))*(H-T-B)}}
 var h='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var yt=S.log?[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.25,1.5,2,3,4,6,8].filter(function(v){return Math.log(v)>=lo&&Math.log(v)<=hi}):niceTicks(lo,hi);
 yt.forEach(function(v){h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(v)+'" y2="'+sy(v)+'" class=grid /><text x="'+(L-5)+'" y="'+(sy(v)+3)+'" text-anchor=end class=ax>'+v+'</text>'});
 var xt=[[zx,P.axis0||""]].concat(Array.from(new Set(ns)).sort(function(a,b){return a-b}).map(function(n){return [n,""+n]}));
 xt.forEach(function(t){h+='<line x1="'+sx(t[0])+'" x2="'+sx(t[0])+'" y1="'+T+'" y2="'+(H-B)+'" class=grid /><text x="'+sx(t[0])+'" y="'+(H-B+14)+'" text-anchor=middle class=ax>'+t[1]+'</text>'});
 h+='<text x="'+(W/2)+'" y="'+(H-6)+'" text-anchor=middle class=ax>simulations (log2)</text>';
 P.pending.forEach(function(p){if(p.n<=nmax*1.18&&p.n>=zx){h+='<line x1="'+sx(p.n)+'" x2="'+sx(p.n)+'" y1="'+T+'" y2="'+(H-B)+'" class=mark /><text x="'+(sx(p.n)+3)+'" y="'+(T+10)+'" class=ax>n'+p.n+' scoring</text>'}});
 if(showRw)SERS.forEach(function(b){if(ROLE[b[0]]=="baseline"&&SEL().indexOf(b[0])>=0)h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(1)+'" y2="'+sy(1)+'" stroke="'+P.col[b[0]]+'" stroke-width=2 ><title>'+b[1]+' = 1</title></line>'});
 var any=false,sum=[],drawn={};
 lv.forEach(function(l,li){var mk=MK[li];
  RUNK.concat(REFK.filter(function(k){return ROLE[k]!="baseline"})).forEach(function(s){
   if(SEL().indexOf(s)<0)return;
   var lab=SERS.find(function(x){return x[0]==s})[1],c=P.col[s];
   vs.forEach(function(v,vi){
    var rowsV=rowsFor(v);
    var rs=rowsV.filter(function(r){return r[1]==l[0]&&r[3]==s}).sort(function(a,b){return a[4]-b[4]});
    if(!rs.length)return;any=true;
    var vtag=vs.length>1?" ["+v+"]":"",dash=vs.length>1&&VDASH[vi%4]?' stroke-dasharray="'+VDASH[vi%4]+'"':"";
    if(ROLE[s]){var val=rs[0][6],key=l[0]+s+val;if(drawn[key])return;drawn[key]=1;
     h+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+sy(val)+'" y2="'+sy(val)+'" stroke="'+c+'" stroke-width=1.5 stroke-dasharray="'+(ROLE[s]=="start"?"6 3":"2 3")+'" />';
     h+=mark(mk,sx(zx),sy(val),c,l[1]+", "+lab+": "+fmt(val))}
    else{
     var zs=rowsV.filter(function(r){return r[1]==l[0]&&ROLE[r[3]]=="start"})[0];
     var pts=(zs?[[zx,zs[6]]]:[]).concat(rs.map(function(r){return [r[4],r[6]]}));
     h+='<polyline points="'+pts.map(function(p){return sx(p[0])+","+sy(p[1])}).join(" ")+'" fill=none stroke="'+c+'" stroke-width=1.8'+dash+' />';
     rs.forEach(function(r){h+=mark(mk,sx(r[4]),sy(r[6]),c,l[1]+", "+lab+vtag+" n"+r[4]+" "+r[5]+": "+fmt(r[6]));
      if(s==RUNK[0]&&lv.length==1&&vs.length==1)h+='<text x="'+(sx(r[4])+6)+'" y="'+(sy(r[6])-6)+'" class=ax>'+fmt(r[6])+'</text>'});
     if(RUNK.indexOf(s)>=0){var last=rs[rs.length-1];
      sum.push((lv.length>1?l[1]+": ":"")+lab+vtag+" n"+last[4]+": "+fmt(last[6])+(zs?" vs zero-shot "+fmt(zs[6])+", "+((last[6]/zs[6]-1)*100>=0?"+":"")+((last[6]/zs[6]-1)*100).toFixed(0)+"%":""))}}})})});
 h+='</svg>';
 var leg=SERS.filter(function(s){return SEL().indexOf(s[0])>=0}).map(function(s){return '<span><i style="background:'+P.col[s[0]]+'"></i>'+s[1]+(ROLE[s[0]]=="start"?" (dashed)":ROLE[s[0]]=="other"?" (dotted)":"")+'</span>'}).join("");
 if(vs.length>1)leg+=vs.map(function(v,i){return '<span><svg width=26 height=8 viewBox="0 0 26 8"><line x1=0 x2=26 y1=4 y2=4 stroke="var(--fg)" stroke-width=2'+(VDASH[i%4]?' stroke-dasharray="'+VDASH[i%4]+'"':"")+' /></svg>'+v+'</span>'}).join("");
 if(lv.length>1)leg+=lv.map(function(l,i){return '<span><svg width=12 height=12 viewBox="0 0 12 12">'+mark(MK[i],6,6,"var(--fg)","")+'</svg>'+l[1]+'</span>'}).join("");
 var note="";
 REFK.forEach(function(k){if(ROLE[k]!="baseline"&&SEL().indexOf(k)>=0&&!rows.some(function(r){return r[3]==k})){var s=SERS.find(function(x){return x[0]==k});note+="<p class=mut>"+(s?s[1]:k)+" pending</p>"}});
 if(P.pending.length)note+='<p class=mut>Saved, not yet in the scores: '+P.pending.map(function(p){return p.run+" n"+p.n+(p.hash?" "+p.hash:"")}).join(", ")+'.</p>';
 if(!any)note+="<p class=mut>No rows for this selection.</p>";
 $("splot").innerHTML='<div class=sum><span class=mut>rel. '+MM.toUpperCase()+' vs randomwalknowcast</span>'+ftInfo("perf")+' '+(sum.join("; ")||"")+'</div>'+h+'<div class=leg>'+leg+'</div>'+note;
}
window.addEventListener("ftsel",draw);
draw();
})();
