import {FaceLandmarker, FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const WASM  = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const $ = id => document.getElementById(id);

/* ---------- 21 MediaPipe landmarks ---------- */
// brows L(inner,mid,outer) R(inner,mid,outer) | lids L(up,lo) R(up,lo) | eye outer L,R |
// nose tip | mouth L,R,upper,lower | chin | forehead | cheeks L,R
const LM21 = [107,105,46, 336,334,276, 159,145,386,374, 33,263, 1, 61,291,13,14, 152, 10, 205,425];

/* ---------- expressions ---------- */
const EX = {
  happy:['😄','Happiness'], sad:['😢','Sadness'], fear:['😱','Fear'],
  disgust:['🫤','Disgust'], surprise:['😲','Surprise'], angry:['😠','Angry']
};
const KEYS = Object.keys(EX);

/* ---------- features (normalised by inter-ocular distance) ---------- */
function features(P){
  const d=(a,b)=>Math.hypot(P(a).x-P(b).x, P(a).y-P(b).y), iod=d(33,263);
  return {
    brow : (d(105,159)+d(334,386))/2/iod,
    inner: (d(107,159)+d(336,386))/2/iod,
    outer: (d(46,33)+d(276,263))/2/iod,
    bdist: d(107,336)/iod,
    eye  : (d(159,145)+d(386,374))/2/iod,
    mopen: d(13,14)/iod,
    mwid : d(61,291)/iod,
    lift : ((P(13).y+P(14).y)/2-(P(61).y+P(291).y)/2)/iod,
    cheek: (d(205,145)+d(425,374))/2/iod,
    nlip : d(1,13)/iod,
    yaw  : (P(1).x-(P(33).x+P(263).x)/2)/iod
  };
}

/* ---------- classifier: deltas from neutral baseline -> 7-way softmax ---------- */
// Swap this function for the trained TSL MLP (export weights to JSON) if desired.
function classify(d){
  const m = v => Math.max(0, v);
  const s = {
    neutral : 1.2,
    happy   : m(d.lift/.03 + d.mwid/.08 - d.cheek/.03),
    sad     : m(-d.lift/.025 + (d.inner-d.outer)/.03),
    surprise: m(d.brow/.04 + d.eye/.025 + d.mopen/.08 - d.mwid/.1),
    fear    : m(d.brow/.05 + d.eye/.03 - d.bdist/.05 + d.mwid/.08),
    angry   : m(-d.inner/.03 - d.bdist/.04 - d.eye/.02),
    disgust : m(-d.nlip/.02 - d.inner/.05 - d.lift/.05)
  };
  const z = Object.values(s).map(Math.exp), Z = z.reduce((a,b)=>a+b,0);
  let best='neutral', bp=0;
  Object.keys(s).forEach((k,i)=>{ if(z[i]/Z>bp){bp=z[i]/Z;best=k} });
  return {label:best, p:bp};
}

/* ---------- state ---------- */
let landmarker, audio;
let level=1, total=0, seq=[], bpm=60, beatMs=1000, t0=0, running=false, cur=-1;
let base=null, calib=null, ema=null, last={label:'neutral',p:0}, frontal=true;
let holdT=null, hit=[], res=[];
const video=$('v'), cv=$('c'), ctx=cv.getContext('2d');

/* ---------- audio ---------- */
function tick(hi){
  if(!audio) return;
  const o=audio.createOscillator(), g=audio.createGain();
  o.frequency.value=hi?880:660; g.gain.value=.15;
  o.connect(g); g.connect(audio.destination); o.start(); o.stop(audio.currentTime+.06);
}


/* ---------- metronome on the audio clock ---------- */
const metro = {timer:null, n:0, vol:.02, beatsPerBar:4};

/* ---------- audio assets (assets/*.wav, assets/music.mp3 | music.wav) ---------- */
const SND = {};
const MUSIC = {bpm:185, firstBeat:.35, gain:.6};   // measure bpm / first-beat time (s) of your real track
let musicSrc=null, musicGain=null;

async function loadBuf(name, urls){
  for(const u of [].concat(urls)){
    try{
      const r=await fetch(u); if(!r.ok) continue;
      SND[name]=await audio.decodeAudioData(await r.arrayBuffer()); return true;
    }catch(e){}
  }
  return false;                                  // falls back to synth click / no music
}
async function loadSounds(){
  await Promise.all([
    loadBuf('hi',   'assets/click_hi.wav'),
    loadBuf('lo',   'assets/click_lo.wav'),
    loadBuf('music',['assets/music.m4a','assets/music.wav'])
  ]);
}

function clickAt(when, accent){
  const g=audio.createGain(); g.connect(audio.destination);
  const buf=SND[accent?'hi':'lo'];
  if(buf){                                       // sample click
    const src=audio.createBufferSource(); src.buffer=buf; g.gain.value=metro.vol;
    src.connect(g); src.start(when);
  }else{                                         // synth fallback
    const o=audio.createOscillator(); o.type='square'; o.frequency.value=accent?1000:700;
    g.gain.setValueAtTime(metro.vol*(accent?1:.6),when);
    g.gain.exponentialRampToValueAtTime(.0001,when+.05);
    o.connect(g); o.start(when); o.stop(when+.06);
  }
}

function startMusic(startPerf){                  // same startPerf as startMetro
  stopMusic(); if(!SND.music) return;
  const when=audio.currentTime+Math.max(0,(startPerf-performance.now())/1000);
  musicSrc=audio.createBufferSource(); musicSrc.buffer=SND.music; musicSrc.loop=true;
  musicSrc.playbackRate.value=bpm/MUSIC.bpm;     // stretch track to level BPM
  musicGain=audio.createGain(); musicGain.gain.value=metro.vol?MUSIC.gain:0;
  musicSrc.connect(musicGain); musicGain.connect(audio.destination);
  musicSrc.start(when, MUSIC.firstBeat);
}
function stopMusic(){ try{musicSrc&&musicSrc.stop()}catch(e){} musicSrc=null; }

// startPerf = performance.now() time (ms) of the first click
function startMetro(startPerf){
  stopMetro(); metro.n = 0;
  metro.timer = setInterval(()=>{
    const horizon = performance.now() + 120;     // schedule 120 ms ahead
    while(startPerf + metro.n*beatMs < horizon){
      const tPerf = startPerf + metro.n*beatMs;
      const when  = audio.currentTime + Math.max(0,(tPerf - performance.now())/1000);
      clickAt(when, metro.n % metro.beatsPerBar === 0);
      metro.n++;
    }
  }, 25);
}
const stopMetro = () => clearInterval(metro.timer);

/* ---------- sequence / hints ---------- */
const ROUNDS = 2, SEQ_LEN = 8;
let round = 1, levelRes = [];   

function makeSeq(){
  const out = [];
  while(out.length < SEQ_LEN){
    const k = KEYS[Math.random()*KEYS.length | 0];
    if(k !== out[out.length-1]) out.push(k);     // no same expression twice in a row
  }
  return out; 
}

const HIDE_FROM_LEVEL = 10;
const hintMode = () => level < HIDE_FROM_LEVEL ? 'all' : 'cur';

function drawTrack(){
  const t=$('track'); t.innerHTML='';
  seq.forEach((k,i)=>{
    const d=document.createElement('div'), m=hintMode();
    const show = m==='all' || (m==='next' && (i===cur||i===cur+1)) || (m==='cur' && i===cur);
    d.className='cell'+(i===cur?' cur':'');
    d.textContent = (res[i]!==undefined||show) ? EX[k][0] : '?';
    if(res[i]!==undefined) d.classList.add(res[i]===100?'g':res[i]>0?'y':'r');
    t.appendChild(d);
  });
}
const border = c => $('stage').style.borderColor=c;

/* ---------- recognition loop ---------- */
let lastVT=-1;
function loop(){
  if(video.readyState>=2 && video.currentTime!==lastVT){
    lastVT=video.currentTime;
    const r=landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0,0,cv.width,cv.height);
    if(r.faceLandmarks.length){
      const L=r.faceLandmarks[0], W=video.videoWidth, H=video.videoHeight;
      const P=i=>({x:L[i].x*W, y:L[i].y*H});
      ctx.fillStyle='#0f0';
      LM21.forEach(i=>{ctx.beginPath();ctx.arc(P(i).x,P(i).y,3,0,7);ctx.fill()});
      const f=features(P);
      ema = ema ? Object.fromEntries(Object.keys(f).map(k=>[k,.5*ema[k]+.5*f[k]])) : f;
      frontal = Math.abs(ema.yaw)<.25;
      if(calib){ calib.push({...ema}); }
      else if(base){
        const d=Object.fromEntries(Object.keys(base).map(k=>[k,ema[k]-base[k]]));
        last=classify(d);
        $('det').textContent = frontal ? `${last.label} ${(last.p*100)|0}%` : 'face front';
      }
      const now=performance.now();
      if(running && cur>=0 && hit[cur]===undefined){
        if(frontal && last.label===seq[cur] && last.p>.5){
          if(holdT===null) holdT=now;
          if(now-holdT>=150) hit[cur]=holdT;   // held >=150 ms
        } else holdT=null;
      }
    } else $('det').textContent='no face';
  }
  requestAnimationFrame(loop);
}

/* ---------- scoring ---------- */
function score(i){
  if(hit[i]===undefined) return 0;
  const d=(hit[i]-(t0+i*beatMs))/beatMs;
  return d<=.25?100 : d<=.6?70 : 50;
}
function finish(i){
  if(i<0||res[i]!==undefined) return;
  const s=res[i]=score(i); total+=s; $('score').textContent=total;
  border(s===100?'#2ecc71':s>0?'#f1c40f':'#e74c3c');
  $('msg').textContent = s===100?'Perfect! +100':s===70?'Late +70':s===50?'Off-beat +50':'Wrong 0';
  drawTrack();
}

/* ---------- game loop ---------- */
function frame(){
  if(!running) return;
  const idx=Math.floor((performance.now()-t0)/beatMs+.25);   // window opens .25 beat early
  if(idx>=seq.length){ finish(cur); return end(); }
  if(idx!==cur){
    finish(cur); cur=idx; holdT=null;
    $('target').textContent=EX[seq[cur]][0]; $('tname').textContent=EX[seq[cur]][1];
    border('#fff'); drawTrack();
  }
  requestAnimationFrame(frame);
}
function end(){
  stopMetro(); stopMusic();
  running = false; cur = -1;
  levelRes.push(...res);                         // keep this round's 8 scores
  $('target').textContent = ''; $('tname').textContent = '';

  if(round < ROUNDS){                            // go straight to round 2
    round++;
    $('msg').textContent = 'Round clear! Get ready…';
    setTimeout(start, 5);
    return;
  }

  const avg = levelRes.reduce((a,b)=>a+b,0) / levelRes.length;   // average over all 16
  $('msg').textContent = `Level ${level} done! Avg ${avg.toFixed(0)}`;
  level++; round = 1; levelRes = [];             // always advance, no retry
  $('btn').disabled = false; $('recal').disabled = false;
  $('btn').textContent = 'Next level';
}
const wait = ms => new Promise(r=>setTimeout(r,ms));

async function calibrate(){
  $('msg').textContent='Neutral face…'; calib=[]; await wait(2000);
  const s=calib; calib=null;
  if(s.length<10){ $('msg').textContent='No face found'; return false; }
  base=Object.fromEntries(Object.keys(s[0]).map(k=>[k,s.reduce((a,b)=>a+b[k],0)/s.length]));
  return true;
}
const sleepUntil = t => wait(Math.max(0, t - performance.now()));

async function start(){
  $('btn').disabled=true; $('recal').disabled=true;
  audio = audio || new AudioContext();
  if(audio.state==='suspended') await audio.resume();0
  if(!SND.music && !SND.hi) await loadSounds();
  if(!base && !(await calibrate())){ $('btn').disabled=false; $('recal').disabled=false; return; }

  const BPM_START = 185, BPM_STEP = 5, BPM_MAX = 250;
  bpm = Math.min(BPM_START + (level-1)*BPM_STEP, BPM_MAX); beatMs = 60000/bpm;

  seq=makeSeq(); res=[]; hit=[]; cur=-1;
  $('lvl').textContent=level; $('rnd').textContent=round+'/'+ROUNDS;
  $('bpm').textContent=bpm; $('msg').textContent=''; border('#333'); drawTrack();

  const COUNT_IN = 8;                            // single source of truth
  const t1 = performance.now() + 200;            // first count-in click
  t0 = t1 + COUNT_IN*beatMs;                     // beat 0 starts right after "1"
  startMetro(t1);
  startMusic(t1);

  for(let i=0;i<COUNT_IN;i++){
    await sleepUntil(t1 + i*beatMs);             // wait for the click
    $('target').textContent = COUNT_IN - i;      // 4, 3, 2, 1
  }

  await sleepUntil(t0 - beatMs*.25);             // hit window opens .25 beat early
  $('target').textContent = '';
  running = true; frame();
}

/* ---------- init ---------- */
(async()=>{
  try{
    const fs=await FilesetResolver.forVisionTasks(WASM);
    landmarker=await FaceLandmarker.createFromOptions(fs,{
      baseOptions:{modelAssetPath:MODEL, delegate:'GPU'},
      runningMode:'VIDEO', numFaces:1
    });
    video.srcObject=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
    await new Promise(r=>video.onloadedmetadata=r);
    cv.width=video.videoWidth; cv.height=video.videoHeight;
    $('btn').disabled=false; $('recal').disabled=false; $('btn').textContent='Start';
    $('btn').onclick=start;
    $('mute').onclick=()=>{metro.vol=metro.vol?0:.25; if(musicGain)musicGain.gain.value=metro.vol?MUSIC.gain:0; $('mute').textContent=metro.vol?'🔊':'🔇'}; $('recal').onclick=()=>{base=null; $('msg').textContent='Will recalibrate on Start'};
    loop();
  }catch(e){ $('status').textContent='Error: '+e.message; }
})();
