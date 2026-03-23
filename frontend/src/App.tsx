import { Routes, Route, Link, useLocation } from "react-router-dom";
import Upload from "./pages/Upload";
import Jobs from "./pages/Jobs";
import Models from "./pages/Models";

const navItems = [
  { to: "/", label: "Загрузка", icon: "↑" },
  { to: "/jobs", label: "Задачи", icon: "◷" },
  { to: "/models", label: "Модели", icon: "⬇" },
];

function NavLink({ to, label, icon }: { to: string; label: string; icon: string }) {
  const loc = useLocation();
  const active = loc.pathname === to || (to !== "/" && loc.pathname.startsWith(to));
  return (
    <Link
      to={to}
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--space-3)",
        padding: "var(--space-3) var(--space-4)",
        borderRadius: "var(--radius-md)",
        color: active ? "var(--accent)" : "rgba(255,255,255,0.8)",
        textDecoration: "none",
        fontWeight: active ? 600 : 500,
        fontSize: "14px",
        background: active ? "rgba(255,255,255,0.08)" : "transparent",
        transition: "all var(--transition)",
      }}
    >
      <span style={{ fontSize: "1.1em" }}>{icon}</span>
      {label}
    </Link>
  );
}

function App() {
  return (
    <div
      className="app-layout"
      style={{
        display: "flex",
        minHeight: "100vh",
      }}
    >
      <aside
        style={{
          width: 240,
          background: "var(--bg-sidebar)",
          color: "var(--text-inverse)",
          padding: "var(--space-6) var(--space-4)",
          display: "flex",
          flexDirection: "column",
          gap: "var(--space-2)",
        }}
      >
        <Link
          to="/"
          style={{
            color: "inherit",
            textDecoration: "none",
            fontFamily: "var(--font-heading)",
            fontWeight: 700,
            fontSize: "1.2rem",
            letterSpacing: "-0.03em",
            marginBottom: "var(--space-6)",
          }}
        >
          SimpleAutoML
        </Link>
        {navItems.map((item) => (
          <NavLink key={item.to} to={item.to} label={item.label} icon={item.icon} />
        ))}
      </aside>
      <main
        style={{
          flex: 1,
          padding: "var(--space-10) var(--space-8)",
          maxWidth: 680,
          margin: "0 auto",
          width: "100%",
        }}
      >
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/models" element={<Models />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
