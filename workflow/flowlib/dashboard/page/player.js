window.timePlayer=function(o){
 var P={playing:false},timer=null;
 function sync(){o.btn.textContent=P.playing?"Pause":"Play";
  window.ftPlayers=window.ftPlayers||{};window.ftPlayers[o.id]=P.playing;
  window.ftPlaying=Object.keys(window.ftPlayers).some(function(k){return window.ftPlayers[k]})}
 function rate(){return +(o.speed&&o.speed.value)||1}
 function arm(){clearInterval(timer);timer=setInterval(function(){o.set((o.get()+1)%o.len())},1000/rate())}
 P.play=function(){if(P.playing)return;P.playing=true;sync();arm()};
 P.pause=function(){if(!P.playing)return;P.playing=false;clearInterval(timer);sync()};
 o.btn.onclick=function(){P.playing?P.pause():P.play()};
 if(o.speed)o.speed.onchange=function(){if(P.playing)arm()};
 sync();return P};
