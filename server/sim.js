'use strict';
const fs   = require('fs');
const path = require('path');

// ── CONFIG ──────────────────────────────────────────────────────
const SPHERE_R      = 200;
const PLANT_INIT    = 80;
const PLANT_MAX     = 120;
const CREATURE_INIT = 8;
const MAX_CREATURES = 20;

const CFG = {
  plant:  { layers:[3,5], baseLen:[2,7], ratio:[0.55,0.70], spread:[1.0,1.4] },
  green:  { hue:[85,135],  sat:[40,75],  light:[22,50]  },
  flower: { hue:[0,360],   sat:[70,100], light:[60,88], count:[2,8], size:[0.4,1.0] },
};

// ── UTILS ────────────────────────────────────────────────────────
const rand    = (a,b) => Math.random()*(b-a)+a;
const randInt = (a,b) => Math.floor(rand(a,b+1));
const pick    = ([a,b]) => rand(a,b);
const pickI   = ([a,b]) => randInt(a,b);
const clamp   = (v,lo,hi) => Math.max(lo, Math.min(hi,v));

// ── 3D VECTOR MATH ───────────────────────────────────────────────
const vNorm  = v => { const l=Math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z); return l<1e-9?{x:0,y:0,z:1}:{x:v.x/l,y:v.y/l,z:v.z/l}; };
const vCross = (a,b) => ({x:a.y*b.z-a.z*b.y, y:a.z*b.x-a.x*b.z, z:a.x*b.y-a.y*b.x});
function rodrigues(v,k,angle){
  const cos=Math.cos(angle),sin=Math.sin(angle),dot=v.x*k.x+v.y*k.y+v.z*k.z,cr=vCross(k,v);
  return {x:v.x*cos+cr.x*sin+k.x*dot*(1-cos), y:v.y*cos+cr.y*sin+k.y*dot*(1-cos), z:v.z*cos+cr.z*sin+k.z*dot*(1-cos)};
}
function randomPerp(d){
  const ref=Math.abs(d.x)<0.9?{x:1,y:0,z:0}:{x:0,y:1,z:0};
  return rodrigues(vNorm(vCross(d,ref)), d, rand(0,Math.PI*2));
}
const deviate   = (d,a) => vNorm(rodrigues(d,randomPerp(d),a));
const randomDir = () => { const t=Math.acos(2*Math.random()-1),p=rand(0,Math.PI*2); return {x:Math.sin(t)*Math.cos(p),y:Math.cos(t),z:Math.sin(t)*Math.sin(p)}; };
function randomInsideSphere(r){
  const rr=Math.cbrt(Math.random())*r, t=Math.acos(2*Math.random()-1), p=rand(0,Math.PI*2);
  return {x:rr*Math.sin(t)*Math.cos(p), y:rr*Math.sin(t)*Math.sin(p), z:rr*Math.cos(t)};
}
const dist3 = (a,b) => Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2);

// ── PLANT SYSTEM ─────────────────────────────────────────────────
function buildPlantGeometry(root){
  const layers=pickI(CFG.plant.layers),baseLen=pick(CFG.plant.baseLen),
        ratio=pick(CFG.plant.ratio),spread=pick(CFG.plant.spread),
        gh=pick(CFG.green.hue),gs=pick(CFG.green.sat),gl=pick(CFG.green.light),
        fh=pick(CFG.flower.hue),fs=pick(CFG.flower.sat),fl=pick(CFG.flower.light),
        fc=pickI(CFG.flower.count),fsz=pick(CFG.flower.size);
  const segs=[],flowers=[],tips=[];
  const rec=(px,py,pz,dir,len,depth)=>{
    const ex=px+dir.x*len,ey=py+dir.y*len,ez=pz+dir.z*len,t=depth/layers;
    segs.push({x1:px,y1:py,z1:pz,x2:ex,y2:ey,z2:ez,color:`hsl(${gh},${gs}%,${gl+(1-t)*18}%)`,lw:0.2+depth*0.2});
    if(depth===1){tips.push({x:ex,y:ey,z:ez});return;}
    const h=spread*0.5;
    rec(ex,ey,ez,deviate(dir,h+rand(-0.06,0.06)),len*ratio,depth-1);
    rec(ex,ey,ez,deviate(dir,h+rand(-0.06,0.06)),len*ratio,depth-1);
  };
  rec(root.x,root.y,root.z,randomDir(),baseLen,layers);
  tips.sort(()=>Math.random()-0.5).slice(0,Math.min(fc,tips.length))
      .forEach(t=>flowers.push({x:t.x,y:t.y,z:t.z,color:`hsl(${fh+rand(-22,22)},${fs}%,${fl}%)`,size:fsz}));
  return {segs,flowers};
}
function createPlantAt(root){
  const {segs,flowers}=buildPlantGeometry(root);
  return {segs,flowers,root,alive:true,regrowTimer:0,energyValue:rand(25,40),idx:0};
}
function regeneratePlant(pl){
  const {segs,flowers}=buildPlantGeometry(pl.root);
  pl.segs=segs; pl.flowers=flowers; pl.alive=true; pl.regrowTimer=0;
  plantDirty.add(pl.idx);
}

const plantObjects = [];
const plantDirty   = new Set();

// ── NEURAL NETWORK ───────────────────────────────────────────────
class NeuralNet {
  constructor(weights){
    if(weights){this.w1=new Float32Array(weights.slice(0,96));this.w2=new Float32Array(weights.slice(96,128));}
    else{this.w1=Float32Array.from({length:96},()=>rand(-1,1));this.w2=Float32Array.from({length:32},()=>rand(-1,1));}
  }
  forward(inp){
    const h=new Float32Array(8);
    for(let j=0;j<8;j++){for(let i=0;i<12;i++)h[j]+=inp[i]*this.w1[i*8+j];h[j]=Math.tanh(h[j]);}
    const o=new Float32Array(4);
    for(let k=0;k<4;k++){for(let j=0;j<8;j++)o[k]+=h[j]*this.w2[j*4+k];o[k]=Math.tanh(o[k]);}
    return o;
  }
}

// ── GENOME + GA ──────────────────────────────────────────────────
function makeRandomGenome(){
  return {weights:Float32Array.from({length:128},()=>rand(-1,1)),bodyRadius:rand(4,12),legLength:rand(5,16),hue:rand(160,220),maturityBias:rand(0.7,1.3)};
}
function crossoverGenome(gA,gB,mutRate=0.05){
  const w=new Float32Array(128);
  for(let i=0;i<128;i++){
    w[i]=Math.random()<0.5?gA.weights[i]:gB.weights[i];
    if(Math.random()<mutRate) w[i]=clamp(w[i]+rand(-0.3,0.3),-3,3);
  }
  return {weights:w,bodyRadius:clamp((gA.bodyRadius+gB.bodyRadius)/2+rand(-0.6,0.6),3,12),legLength:clamp((gA.legLength+gB.legLength)/2+rand(-1,1),4,16),hue:clamp((gA.hue+gB.hue)/2+rand(-6,6),140,240),maturityBias:clamp((gA.maturityBias+gB.maturityBias)/2+rand(-0.06,0.06),0.7,1.3)};
}

// ── GENE POOL + ERA TRACKING ─────────────────────────────────────
const genePool        = [];
const eraHistory      = [];
let   eraFrame        = 0;
let   currentEraStats = { era:1, peakPop:0, maxGen:0 };

function deepCopyGenome(g){
  return {weights:new Float32Array(g.weights),bodyRadius:g.bodyRadius,legLength:g.legLength,hue:g.hue,maturityBias:g.maturityBias};
}
function mutateGenome(g,rate=0.15,scale=0.5){
  const w=new Float32Array(128);
  for(let i=0;i<128;i++) w[i]=g.weights[i]+(Math.random()<rate?rand(-scale,scale):0);
  return {weights:w,bodyRadius:clamp(g.bodyRadius+rand(-1.5,1.5),3,12),legLength:clamp(g.legLength+rand(-2,2),4,16),hue:clamp(g.hue+rand(-15,15),140,240),maturityBias:clamp(g.maturityBias+rand(-0.1,0.1),0.7,1.3)};
}
function recordFitness(creature){
  genePool.push({genome:deepCopyGenome(creature.genome),fitness:creature.age,era:currentEraStats.era});
  genePool.sort((a,b)=>b.fitness-a.fitness);
  if(genePool.length>8) genePool.pop();
}
function seedGenome(){
  if(genePool.length>=2){
    const iA=Math.floor(Math.random()*Math.min(4,genePool.length));
    const iB=Math.floor(Math.random()*genePool.length);
    return crossoverGenome(genePool[iA].genome,genePool[iB].genome,0.08);
  }
  if(genePool.length===1) return mutateGenome(genePool[0].genome);
  return makeRandomGenome();
}

// ── CREATURE CLASS ───────────────────────────────────────────────
class Creature {
  constructor(genome,pos,generation=0){
    this.genome=genome;
    this.pos=pos||randomInsideSphere(SPHERE_R*0.7);
    this.dir=randomDir();
    this.generation=generation;
    this.bodyRadius=genome.bodyRadius;
    this.legLength=genome.legLength;
    this.maxSpeed=0.25+genome.legLength*0.055;
    this.energyDrain=0.016+genome.bodyRadius*0.0028;
    this.nn=new NeuralNet(genome.weights);
    this.energy=70;
    this.age=0;
    this.maturityAge=Math.round(360*genome.maturityBias);
    this.maxAge=7200+randInt(-600,600);
    this.state='juvenile';
    this.scale=0.4;
    this.alpha=0;
    this.fadingIn=true;
    this.matingCooldown=0;
    this.mateIntent=0;
    this.legAngle=rand(0,Math.PI*2);
  }

  update(plants,creatures){
    if(this.fadingIn){this.alpha=Math.min(1,this.alpha+0.025);if(this.alpha>=1)this.fadingIn=false;}
    if(this.state==='dying'){this.alpha-=0.02;if(this.alpha<=0)this.state='dead';return;}

    this.age++;
    if(this.matingCooldown>0) this.matingCooldown--;

    if(this.state==='juvenile'){
      this.scale=Math.min(1,0.4+0.6*(this.age/this.maturityAge));
      if(this.age>=this.maturityAge) this.state='adult';
    }

    this.energy-=this.energyDrain;
    if(this.energy<=0||this.age>=this.maxAge){ recordFitness(this); this.state='dying'; return; }

    const inp=this._getInputs(plants,creatures);
    const out=this.nn.forward(inp);
    this.mateIntent=out[3];

    this.dir=vNorm(rodrigues(this.dir,{x:0,y:1,z:0},out[0]*0.08));
    const refUp=Math.abs(this.dir.y)<0.9?{x:0,y:1,z:0}:{x:1,y:0,z:0};
    this.dir=vNorm(rodrigues(this.dir,vNorm(vCross(this.dir,refUp)),out[1]*0.05));

    const speed=((out[2]+1)/2)*this.maxSpeed;
    this.pos.x+=this.dir.x*speed;
    this.pos.y+=this.dir.y*speed;
    this.pos.z+=this.dir.z*speed;

    const d=Math.sqrt(this.pos.x**2+this.pos.y**2+this.pos.z**2);
    if(d>SPHERE_R*0.90){
      const n=vNorm(this.pos),dot=this.dir.x*n.x+this.dir.y*n.y+this.dir.z*n.z;
      this.dir.x-=2*dot*n.x;this.dir.y-=2*dot*n.y;this.dir.z-=2*dot*n.z;
      this.dir=vNorm(this.dir);
      this.pos.x=n.x*SPHERE_R*0.87;this.pos.y=n.y*SPHERE_R*0.87;this.pos.z=n.z*SPHERE_R*0.87;
    }

    const eatR=this.bodyRadius*this.scale+5;
    for(const pl of plants){
      if(!pl.alive) continue;
      if(dist3(this.pos,pl.root)<eatR){pl.alive=false;pl.regrowTimer=randInt(480,720);this.energy=Math.min(100,this.energy+pl.energyValue);}
    }
    this.legAngle+=speed*0.3;
  }

  _getInputs(plants,creatures){
    let pDist=1,pdx=0,pdy=0,pdz=0,minPD=Infinity;
    for(const pl of plants){
      if(!pl.alive) continue;
      const d=dist3(this.pos,pl.root);
      if(d<minPD){minPD=d;pDist=Math.min(d/(SPHERE_R*2),1);const n=vNorm({x:pl.root.x-this.pos.x,y:pl.root.y-this.pos.y,z:pl.root.z-this.pos.z});pdx=n.x;pdy=n.y;pdz=n.z;}
    }
    let mDist=1,mdx=0,mdy=0,mdz=0,minMD=Infinity;
    for(const c of creatures){
      if(c===this||c.state!=='adult') continue;
      const d=dist3(this.pos,c.pos);
      if(d<minMD){minMD=d;mDist=Math.min(d/(SPHERE_R*2),1);const n=vNorm({x:c.pos.x-this.pos.x,y:c.pos.y-this.pos.y,z:c.pos.z-this.pos.z});mdx=n.x;mdy=n.y;mdz=n.z;}
    }
    return [pDist,pdx,pdy,pdz,mDist,mdx,mdy,mdz,this.energy/100,this.genome.bodyRadius/12,this.age/this.maxAge,(this.matingCooldown<=0&&this.energy>60&&this.state==='adult')?1:0];
  }
}

const creatures = [];

// ── TICK ─────────────────────────────────────────────────────────
let frameCount = 0;

function tick(){
  frameCount++;

  let alivePlantCount=0;
  for(const pl of plantObjects){
    if(pl.alive){alivePlantCount++;}
    else{if(--pl.regrowTimer<=0) regeneratePlant(pl);}
  }
  if(frameCount%180===0&&alivePlantCount<PLANT_MAX){
    const pl=createPlantAt(randomInsideSphere(SPHERE_R*0.92));
    pl.idx=plantObjects.length;
    plantDirty.add(pl.idx);
    plantObjects.push(pl);
  }

  for(const c of creatures) c.update(plantObjects,creatures);

  if(creatures.length<MAX_CREATURES){
    for(let i=0;i<creatures.length;i++){
      for(let j=i+1;j<creatures.length;j++){
        const a=creatures[i],b=creatures[j];
        if(a.state!=='adult'||b.state!=='adult') continue;
        if(a.energy<60||b.energy<60) continue;
        if(a.matingCooldown>0||b.matingCooldown>0) continue;
        if(a.mateIntent<=0||b.mateIntent<=0) continue;
        if(dist3(a.pos,b.pos)>a.bodyRadius+b.bodyRadius+15) continue;
        const gen=Math.max(a.generation,b.generation)+1;
        const mid={x:(a.pos.x+b.pos.x)/2,y:(a.pos.y+b.pos.y)/2,z:(a.pos.z+b.pos.z)/2};
        for(let k=0;k<randInt(1,5)&&creatures.length<MAX_CREATURES;k++){
          const off=randomInsideSphere(12);
          creatures.push(new Creature(crossoverGenome(a.genome,b.genome),{x:mid.x+off.x,y:mid.y+off.y,z:mid.z+off.z},gen));
        }
        a.energy-=30;b.energy-=30;
        a.matingCooldown=600;b.matingCooldown=600;
      }
    }
  }

  for(let i=creatures.length-1;i>=0;i--)
    if(creatures[i].state==='dead') creatures.splice(i,1);

  eraFrame++;
  const _alive=creatures.filter(c=>c.state!=='dead'&&c.state!=='dying').length;
  if(_alive>currentEraStats.peakPop) currentEraStats.peakPop=_alive;
  const _genNow=creatures.length?Math.max(...creatures.map(c=>c.generation)):0;
  if(_genNow>currentEraStats.maxGen) currentEraStats.maxGen=_genNow;

  if(creatures.length===0){
    eraHistory.push({...currentEraStats,duration:eraFrame});
    if(eraHistory.length>20) eraHistory.shift();
    currentEraStats={era:currentEraStats.era+1,peakPop:0,maxGen:0};
    eraFrame=0;
    for(let i=0;i<CREATURE_INIT;i++) creatures.push(new Creature(seedGenome()));
  }
}

// ── STATE EXPORT ─────────────────────────────────────────────────
function _creatureData(c){
  return {
    x:c.pos.x,y:c.pos.y,z:c.pos.z,
    dx:c.dir.x,dy:c.dir.y,dz:c.dir.z,
    energy:c.energy,hue:c.genome.hue,
    bodyRadius:c.bodyRadius,legLength:c.legLength,
    age:c.age,maxAge:c.maxAge,generation:c.generation,
    state:c.state,scale:c.scale,alpha:c.alpha,
    legAngle:c.legAngle,mateIntent:c.mateIntent,
  };
}

function _eraData(){
  return {
    era:{...currentEraStats,eraFrame},
    eraHistory:eraHistory.slice(-10),
    genePool:genePool.map(e=>({fitness:e.fitness,era:e.era})),
  };
}

function getFullState(){
  return {
    creatures:creatures.map(_creatureData),
    plants:plantObjects.map(pl=>({index:pl.idx,root:pl.root,alive:pl.alive,segs:pl.segs,flowers:pl.flowers})),
    ..._eraData(),
  };
}

function getFrameState(){
  const updates=[];
  for(const idx of plantDirty){
    const pl=plantObjects[idx];
    if(pl) updates.push({index:pl.idx,root:pl.root,alive:pl.alive,segs:pl.segs,flowers:pl.flowers});
  }
  plantDirty.clear();
  return {
    creatures:creatures.map(_creatureData),
    plantAlive:plantObjects.map(p=>p.alive),
    plantUpdates:updates,
    ..._eraData(),
  };
}

// ── PERSISTENCE ──────────────────────────────────────────────────
const STATE_FILE = path.join(__dirname, 'state.json');

function saveState(){
  try {
    const data={
      genePool:genePool.map(e=>({
        genome:{weights:Array.from(e.genome.weights),bodyRadius:e.genome.bodyRadius,legLength:e.genome.legLength,hue:e.genome.hue,maturityBias:e.genome.maturityBias},
        fitness:e.fitness,era:e.era
      })),
      eraHistory,currentEraStats,eraFrame,
      plants:plantObjects.map(p=>({root:p.root,alive:p.alive,regrowTimer:p.regrowTimer,energyValue:p.energyValue})),
    };
    fs.writeFileSync(STATE_FILE, JSON.stringify(data));
  } catch(e){ console.error('Save failed:',e.message); }
}

function loadState(){
  try {
    if(!fs.existsSync(STATE_FILE)){ _initFresh(); return; }
    const data=JSON.parse(fs.readFileSync(STATE_FILE,'utf8'));
    if(data.genePool) data.genePool.forEach(e=>{
      genePool.push({genome:{weights:new Float32Array(e.genome.weights),bodyRadius:e.genome.bodyRadius,legLength:e.genome.legLength,hue:e.genome.hue,maturityBias:e.genome.maturityBias},fitness:e.fitness,era:e.era});
    });
    if(data.eraHistory) eraHistory.push(...data.eraHistory);
    if(data.currentEraStats) Object.assign(currentEraStats,data.currentEraStats);
    if(data.eraFrame) eraFrame=data.eraFrame;
    if(data.plants&&data.plants.length){
      data.plants.forEach((p,i)=>{
        const pl=createPlantAt(p.root);
        pl.alive=p.alive; pl.regrowTimer=p.regrowTimer||0; pl.energyValue=p.energyValue||rand(25,40); pl.idx=i;
        plantObjects.push(pl);
      });
    } else { _initFresh(); }
    for(let i=0;i<CREATURE_INIT;i++) creatures.push(new Creature(seedGenome()));
    console.log(`Resumed — Era ${currentEraStats.era}, gene pool: ${genePool.length} entries`);
  } catch(e){
    console.error('Load failed:',e.message,' — starting fresh');
    if(plantObjects.length===0) _initFresh();
  }
}

function _initFresh(){
  for(let i=0;i<PLANT_INIT;i++){
    const pl=createPlantAt(randomInsideSphere(SPHERE_R*0.92));
    pl.idx=i; plantObjects.push(pl);
  }
  for(let i=0;i<CREATURE_INIT;i++) creatures.push(new Creature(makeRandomGenome()));
}

module.exports = { tick, getFullState, getFrameState, loadState, saveState };
