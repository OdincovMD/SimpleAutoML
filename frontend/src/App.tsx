import { Routes, Route, Link, useLocation } from "react-router-dom";
import Upload from "./pages/Upload";
import Jobs from "./pages/Jobs";
import Models from "./pages/Models";
import Inference from "./pages/Inference";

const navItems = [
  { to: "/", label: "Загрузка", icon: "↑" },
  { to: "/jobs", label: "Задачи", icon: "◷" },
  { to: "/models", label: "Модели", icon: "⬇" },
  { to: "/inference", label: "Инференс", icon: "◇" },
];

function NavLink({ to, label, icon }: { to: string; label: string; icon: string }) {
  const loc = useLocation();
  const active = loc.pathname === to || (to !== "/" && loc.pathname.startsWith(to));
  return (
    <Link
      to={to}
      className={`app-nav-link${active ? " app-nav-link--active" : ""}`}
    >
      <span className="app-nav-link__icon" aria-hidden>
        {icon}
      </span>
      {label}
    </Link>
  );
}

function App() {
  const loc = useLocation();
  const jobsWide = loc.pathname === "/jobs" || loc.pathname.startsWith("/jobs");

  return (
    <div
      className="app-layout"
      style={{
        display: "flex",
        minHeight: "100vh",
      }}
    >
      <aside className="app-sidebar">
        <Link to="/" className="app-brand">
          <span className="app-brand__mark" aria-hidden />
          SimpleAutoML
        </Link>
        {navItems.map((item) => (
          <NavLink key={item.to} to={item.to} label={item.label} icon={item.icon} />
        ))}
      </aside>
      <main
        className={jobsWide ? "page-main--wide" : undefined}
        style={{
          flex: 1,
          padding: "var(--space-10) var(--space-8)",
          maxWidth: jobsWide ? 960 : 680,
          margin: "0 auto",
          width: "100%",
        }}
      >
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/models" element={<Models />} />
          <Route path="/inference" element={<Inference />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
