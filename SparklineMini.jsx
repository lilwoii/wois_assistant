// frontend/src/components/SparklineMini.jsx
import React from "react";

const SparklineMini = ({
  data,
  width = 60,
  height = 18,
}) => {
  if (!Array.isArray(data) || data.length < 2) {
    return (
      <div
        style={{
          width,
          height,
          fontSize: "10px",
          color: "#6b7280",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        —
      </div>
    );
  }

  const n = data.length;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (n - 1)) * (width - 2) + 1;
    const y = height - 1 - ((v - min) / range) * (height - 2);
    return `${x},${y}`;
  });

  const rising = data[data.length - 1] >= data[0];
  const strokeColor = rising ? "#22c55e" : "#ef4444";

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block" }}
    >
      <polyline
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.3"
        points={points.join(" ")}
      />
    </svg>
  );
};

export default SparklineMini;
