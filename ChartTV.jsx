import React, { useEffect, useRef } from 'react';

// Fallback simple canvas sparkline
export default function ChartTV({ symbol, overlay }){
  const ref = useRef(null);
  useEffect(()=>{
    const ctx = ref.current.getContext('2d');
    ctx.clearRect(0,0,600,200);
    ctx.fillStyle='#111'; ctx.fillRect(0,0,600,200);
    ctx.fillStyle='#fff'; ctx.fillText(symbol, 10, 15);
    // very simple illustrative chart
    ctx.strokeStyle='#4af'; ctx.beginPath();
    const data = (overlay?.spark || Array.from({length:100}, (_,i)=>100+Math.sin(i/7)*2));
    data.forEach((v,i)=>{
      const x = i * (600/Math.max(1,data.length-1));
      const y = 180 - (v%10)*15;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
  }, [symbol, overlay]);
  return <canvas ref={ref} width={600} height={200} className="w-full rounded bg-black"/>;
}
