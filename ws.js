const WS = process.env.REACT_APP_WS_BASE || 'ws://localhost:8000/ws';

export function openSocket(onMessage){
  const ws = new WebSocket(WS);
  ws.onmessage = (ev)=>{
    try{ const m = JSON.parse(ev.data); onMessage(m); } catch(e){}
  };
  return ws;
}
