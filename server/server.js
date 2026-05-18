'use strict';
const path    = require('path');
const express = require('express');
const { WebSocketServer } = require('ws');
const sim     = require('./sim');

sim.loadState();

const app = express();
app.use(express.static(path.join(__dirname, '..')));

const server = app.listen(process.env.PORT || 3000, () => {
  console.log(`Neural Ecosystem server on port ${process.env.PORT || 3000}`);
});

const wss = new WebSocketServer({ server });

// Simulation loop at 20fps
setInterval(() => {
  sim.tick();
  const state = sim.getFrameState();
  if(wss.clients.size === 0) return;
  const msg = JSON.stringify({ type:'frame', ...state });
  for(const ws of wss.clients)
    if(ws.readyState === 1) ws.send(msg);
}, 50);

// Save to disk every 60s
setInterval(() => {
  sim.saveState();
  console.log('State saved');
}, 60000);

wss.on('connection', ws => {
  console.log(`Client connected (${wss.clients.size} total)`);
  ws.send(JSON.stringify({ type:'init', ...sim.getFullState() }));
  ws.on('close', () => console.log(`Client disconnected (${wss.clients.size} remaining)`));
  ws.on('error', err => console.error('WS error:', err.message));
});

process.on('SIGTERM', () => { sim.saveState(); process.exit(0); });
process.on('SIGINT',  () => { sim.saveState(); process.exit(0); });
