// frontend/src/theme/ThemeContext.jsx
import React, { createContext, useContext, useEffect, useState } from "react";

const defaultTheme = {
  mode: "midnight",
  accent: "#2563eb",
};

const ThemeContext = createContext({
  theme: defaultTheme,
  setTheme: () => {},
});

export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState(defaultTheme);

  // load from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem("woi_theme");
      if (raw) {
        const parsed = JSON.parse(raw);
        setTheme((prev) => ({ ...prev, ...parsed }));
      }
    } catch {
      // ignore
    }
  }, []);

  // persist to localStorage
  useEffect(() => {
    try {
      localStorage.setItem("woi_theme", JSON.stringify(theme));
    } catch {
      // ignore
    }
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);
export { defaultTheme };
