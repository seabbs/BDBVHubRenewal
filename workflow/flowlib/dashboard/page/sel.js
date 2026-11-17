(function(){var D=null,on=[];
 function render(){var el=document.getElementById("selbar");if(!el||!D)return;
  function b(x){return '<button class="tog'+(on.indexOf(x.key)>=0?' on':'')+'" data-k="'+x.key+'" style="border-left:4px solid '+x.color+'">'+x.label+'</button>'}
  el.innerHTML='<b>Show runs:</b>'+ftInfo("runs")+D.runs.map(b).join("")+'<span class=sep></span><span class=mut title="used by the performance and nowcast charts; the loss and attention-bias panels show runs only">references</span>'+D.refs.map(b).join("")+'<span class=mut style="margin-left:10px">applies to every chart</span>';
  el.onclick=function(e){var bt=e.target.closest("button");if(!bt)return;window.ftSel.toggle(bt.dataset.k)}}
 function save(){try{sessionStorage.setItem("ftsel",JSON.stringify(on))}catch(e){}}
 window.ftSel={
  init:function(d){D=d;var s=null;try{s=JSON.parse(sessionStorage.getItem("ftsel")||"null")}catch(e){}
   var all=d.runs.concat(d.refs).map(function(x){return x.key});
   on=(Array.isArray(s)?s:d.defaults.slice()).filter(function(k){return all.indexOf(k)>=0});
   if(!on.length)on=d.defaults.slice();render()},
  get:function(){return on.slice()},
  refsOn:function(){return D.refs.filter(function(r){return on.indexOf(r.key)>=0})},
  toggle:function(k){var i=on.indexOf(k);if(i>=0){if(on.length>1)on.splice(i,1)}else on.push(k);
   save();render();window.dispatchEvent(new Event("ftsel"))}}})();
