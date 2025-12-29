// frontend/src/components/Sparkline.jsx
import React from "react";

const Sparkline = ({ data = [], positive = true }) => {
  if (!Array.isArray(data) || data.length === 0) {
    return <span style={{ fontSize: 11, opacity: 0.6 }}>—</span>;
  }

  const width = 70;
  const height = 22;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1 || 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  });

  const stroke = positive ? "#22c55e" : "#f97316";

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block" }}
    >
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth="1.5"
        points={points.join(" ")}
      />
    </svg>
  );
};

export default Sparkline;
